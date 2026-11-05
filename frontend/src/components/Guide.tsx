import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api'

export default function Guide() {
  const [content, setContent] = useState<string | null>(null)

  useEffect(() => {
    api.content('guide').then((r) => setContent(r.content)).catch(() => {
      setContent('Failed to load guide content.')
    })
  }, [])

  if (!content) {
    return (
      <div className="content-page">
        <div className="loading-spinner" style={{ margin: '48px auto' }} />
      </div>
    )
  }

  return (
    <div className="content-page">
      <h1 className="content-page__title">Offline Evaluation Guide</h1>
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  )
}
