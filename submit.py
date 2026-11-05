import threading
import queue
import time
from pathlib import Path
import traceback
import datetime # Added for timestamping
from tokenizer_evaluate import run_benchmark, generate_model_key

class SubmissionWorker:
    """
    Manages a queue of model names and evaluates them in a background thread.
    Tracks the status (Queued, Running, Failed, Completed) and errors for each model.
    """
    def __init__(self, benchmark_dir: Path, results_dir: Path, benchmark_file_hashes: dict, tokenizer_hash_cache: dict):
        self.benchmark_dir = benchmark_dir
        self.results_dir = results_dir
        self.benchmark_file_hashes = benchmark_file_hashes
        self.tokenizer_hash_cache = tokenizer_hash_cache

        self.task_queue = queue.Queue() # Queue stores only model names to be processed
        self.is_running = False
        self._thread = None
        
        # Central state tracking for all submitted models
        # Key: model_name, Value: {'status': str, 'submission_time': datetime, 'error': str | None}
        self.model_states = {}
        self.state_lock = threading.Lock() # Lock to protect access to model_states

        print("SubmissionWorker initialized.")

    def submit_model(self, model_name: str, revision: str = "main", subfolder: str = None, _target_mode: bool = False, selected_langs: list[str] | None = None):
        """Adds a model name and selected languages to the evaluation queue and updates its state."""
        if not model_name:
            print("Warning: Empty model name submitted.")
            return
        
        submission_time = datetime.datetime.now()
        # Create model_key using centralized function
        model_key = generate_model_key(model_name, revision, subfolder, _target_mode)
            
        with self.state_lock: # Protect state update
            # Update state or add new entry
            self.model_states[model_key] = {
                'model_name': model_name,
                'revision': revision,
                'subfolder': subfolder,
                '_target_mode': _target_mode,
                'langs': selected_langs,
                'status': 'Queued',
                'submission_time': submission_time,
                'error': None
            }
        
        lang_info = "all languages" if selected_langs is None else f"languages: {', '.join(selected_langs)}"
        subfolder_info = f" (subfolder: {subfolder})" if subfolder else ""
        target_mode_info = " (target mode: enabled)" if _target_mode else ""
        print(f"Submitting model to queue: {model_name}{subfolder_info}{target_mode_info} ({lang_info})")
        self.task_queue.put((model_name, revision, subfolder, _target_mode, selected_langs)) # Put tuple in the queue

    def _process_queue(self):
        """The main loop for the worker thread."""
        print("Worker thread started.")
        while self.is_running:
            selected_langs = None # Ensure selected_langs is defined
            model_key = None # Ensure model_name is defined for error handling
            try:
                # Wait indefinitely for a task, but check is_running periodically
                queue_item = self.task_queue.get(timeout=1) # Get tuple (or None)

                # Handle stop signal (None)
                if queue_item is None:
                    print("Worker thread received stop signal.")
                    break

                # Unpack the tuple
                model_name, revision, subfolder, _target_mode, selected_langs = queue_item

                if model_name is None:  # Signal to stop
                    print("Worker thread received stop signal.")
                    break

                # Create model_key using centralized function
                model_key = generate_model_key(model_name, revision, subfolder, _target_mode)

                # Update status to Running
                with self.state_lock:
                    if model_key in self.model_states:
                        self.model_states[model_key]['status'] = 'Running'
                        # Clear previous error if retrying?
                        self.model_states[model_key]['error'] = None 
                    else:
                        # Should not happen if submitted correctly, but handle defensively
                        print(f"Warning: Model {model_key} not found in states when starting run.")
                        self.task_queue.task_done() # Mark task done even if skipped
                        continue # Skip this task
                
                lang_info = "all languages" if selected_langs is None else f"languages: {', '.join(selected_langs)}"
                subfolder_info = f" (subfolder: {subfolder})" if subfolder else ""
                target_mode_info = " (target mode: enabled)" if _target_mode else ""
                print(f"Worker processing model: {model_key}{subfolder_info}{target_mode_info} ({lang_info})")
                current_error = None
                try:
                    # Run the benchmark
                    run_benchmark(
                        model_name=model_name,
                        revision=revision,
                        subfolder=subfolder,
                        _target_mode=_target_mode,
                        benchmark_dir=self.benchmark_dir,
                        langs=selected_langs, # <<< Pass selected languages >>>
                        verbose=True, # Keep verbose for background logging
                        save_results=True, # Ensure results are saved
                        benchmark_file_hashes=self.benchmark_file_hashes,
                        tokenizer_hash_cache=self.tokenizer_hash_cache,
                        upload_results=True # <<< ALWAYS UPLOAD RESULTS FROM WORKER >>>
                    )
                    # If successful, mark as Completed
                    with self.state_lock:
                        if model_key in self.model_states:
                            self.model_states[model_key]['status'] = 'Completed' 
                    print(f"Worker finished processing model: {model_key}")

                except Exception as e:
                    error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}" # More concise + full traceback
                    print(f"Error evaluating model {model_key}:\n{error_message}") # Print full traceback to console
                    current_error = error_message
                    # Mark as Failed and store error
                    with self.state_lock:
                        if model_key in self.model_states:
                           self.model_states[model_key]['status'] = 'Failed'
                           self.model_states[model_key]['error'] = current_error

                finally:
                    self.task_queue.task_done() # Mark task as complete regardless of outcome

            except queue.Empty:
                # Queue was empty during timeout, continue loop to check is_running
                continue
            except Exception as e:
                # Catch unexpected errors in the loop itself (e.g., accessing state dict)
                loop_error_message = f"Unexpected error in worker thread loop: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(loop_error_message)
                # Try to mark the problematic model as failed if we know which one it was
                if model_key:
                    with self.state_lock:
                        if model_key in self.model_states:
                            self.model_states[model_key]['status'] = 'Failed'
                            self.model_states[model_key]['error'] = loop_error_message
                time.sleep(5) # Avoid fast spinning on error

        print("Worker thread stopped.")

    def start(self):
        """Starts the background worker thread."""
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._process_queue, daemon=True) # Use daemon thread
            self._thread.start()
            print("SubmissionWorker thread started.")
        else:
            print("SubmissionWorker thread already running.")

    def stop(self):
        """Signals the worker thread to stop gracefully."""
        if self.is_running:
            print("Stopping SubmissionWorker thread...")
            is_running_local = self.is_running # Read once
            if is_running_local:
                self.is_running = False
                self.task_queue.put(None)  # Send stop signal
        else:
            print("SubmissionWorker thread is not running.")

    def get_status(self):
        """
        Returns the current status including details for all tracked models
        (Queued, Running, Failed, Completed) and identifies the running model.
        """
        all_details = []
        running_model_key = "Idle"
        queued_count = 0

        with self.state_lock: # Protect reading state
            # Make a copy to avoid holding lock while formatting
            current_states = self.model_states.copy()
        
        # Sort by submission time for consistent display order
        sorted_models = sorted(current_states.items(), key=lambda item: item[1]['submission_time'], reverse=True)

        for model_key, state_info in sorted_models:
            status = state_info['status']
            # Format details for display
            all_details.append({
                'model_key': model_key,
                'model_name': state_info['model_name'],
                'revision': state_info['revision'],
                'subfolder': state_info.get('subfolder'),
                '_target_mode': state_info.get('_target_mode', False),
                'submission_time': state_info['submission_time'].strftime("%Y-%m-%d %H:%M:%S"),
                'status': status,
                'error': state_info['error'] # Include error, None if no error
            })
            if status == 'Running':
                running_model_key = state_info['model_name']
            if status == 'Queued':
                queued_count += 1

        return {
            "is_running": self.is_running,
            "current_model_key": running_model_key,
            "queue_size": queued_count, # Only count queued models for the size display
            "all_details": all_details # List of dicts for all tracked models
        }

# Example usage (optional, for testing submit.py directly)
if __name__ == '__main__':
    # Create dummy paths/data for testing
    dummy_benchmark_dir = Path("./dummy_benchmark")
    dummy_results_dir = Path("./dummy_results")
    dummy_benchmark_dir.mkdir(exist_ok=True)
    dummy_results_dir.mkdir(exist_ok=True)
    dummy_hashes = {}
    dummy_cache = {}

    print("Creating dummy SubmissionWorker...")
    worker = SubmissionWorker(dummy_benchmark_dir, dummy_results_dir, dummy_hashes, dummy_cache)
    print("Starting worker...")
    worker.start()

    print("Submitting test models...")
    worker.submit_model("test-model-1")
    worker.submit_model("test-model-2")

    print("Checking status...")
    print(worker.get_status())

    print("Sleeping for a bit...")
    time.sleep(5) # Give worker time to potentially process (if run_benchmark was real)

    print("Checking status again...")
    print(worker.get_status())

    print("Stopping worker...")
    worker.stop()
    print("Worker stopped.") 