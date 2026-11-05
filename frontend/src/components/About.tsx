import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import about from '../../../docs/about.md?raw'

export default function About() {
  return (
    <article className="prose">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{about}</ReactMarkdown>
    </article>
  )
}
