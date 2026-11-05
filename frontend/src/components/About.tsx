import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api'

export default function About() {
  const [md, setMd] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.content('about').then((r) => setMd(r.content)).catch((e: Error) => setError(e.message))
  }, [])
  if (error) return <div className="notice error">{error}</div>
  if (md == null) return <div className="skeleton" style={{ width: '50%' }} />
  return (
    <article className="prose">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{md}</ReactMarkdown>
    </article>
  )
}
