import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api'

export default function About() {
  const [about, setAbout] = useState<string | null>(null)
  const [description, setDescription] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      api.content('description').then((r) => r.content),
      api.content('about').then((r) => r.content),
    ])
      .then(([desc, ab]) => {
        setDescription(desc)
        setAbout(ab)
      })
      .catch(() => {
        setDescription('Failed to load.')
        setAbout('Failed to load.')
      })
  }, [])

  if (!about || !description) {
    return (
      <div className="content-page">
        <div className="loading-spinner" style={{ margin: '48px auto' }} />
      </div>
    )
  }

  return (
    <div className="content-page">
      <h1 className="content-page__title">About the Leaderboard</h1>
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{description}</ReactMarkdown>
      </div>
      <div className="markdown" style={{ marginTop: '32px' }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{about}</ReactMarkdown>
      </div>
    </div>
  )
}
