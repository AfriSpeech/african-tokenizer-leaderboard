import { useState, useRef, useCallback, useMemo } from 'react'
import { createPortal } from 'react-dom'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type SortingFn,
  type Row,
} from '@tanstack/react-table'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { ColumnHeader } from '../types'

interface Props {
  data: Record<string, unknown>[]
  columns: ColumnHeader[]
  loading?: boolean
  isMultilingual?: boolean
  avgColumns?: string[]
  columnDescriptions?: Record<string, string>
}

const DEFAULT_METRIC_DECIMALS = 3

const PERCENTAGE_COLS = new Set([
  'reversible_ratio',
  'unk_ratio',
  'vocab_near_duplicates',
  'byte_fidelity',
  'char_fidelity',
  'freq_hapax_ratio',
  'freq_gini',
  'freq_shannon_efficiency',
  'freq_renyi_efficiency',
  'freq_shannon_efficiency_full',
  'freq_renyi_efficiency_full',
  'freq_percentile_freq',
  'freq_cardinality_ratio',
  'freq_90pct_mass_ratio',
])

const METRIC_DECIMALS: Record<string, number> = {
  fertility: 3,
  compression_bytes: 3,
  parity_en: 3,
  reversible_ratio: 1,
  unk_ratio: 3,
  total_tokens: 0,
  total_words: 0,
  total_chars: 0,
  total_bytes: 0,
  byte_edit_distance_total: 0,
  char_edit_distance_total: 0,
  total_docs: 0,
  reversible_docs_count: 0,
  unk_tokens_count: 0,
  vocab_size: 0,
  vocab_near_duplicates: 1,
  byte_fidelity: 2,
  effective_bytes_per_token: 3,
  ebpb: 3,
  ebpb_full: 3,
  char_fidelity: 2,
  effective_chars_per_token: 3,
  ebpc: 3,
  freq_cardinality: 0,
  freq_cardinality_ratio: 1,
  freq_auc: 3,
  freq_slope: 3,
  freq_power_law: 3,
  freq_entropy: 3,
  freq_renyi_entropy: 3,
  freq_shannon_efficiency: 3,
  freq_renyi_efficiency: 3,
  freq_shannon_efficiency_full: 3,
  freq_renyi_efficiency_full: 3,
  freq_percentile_freq: 3,
  freq_hapax_count: 0,
  freq_hapax_ratio: 1,
  freq_half_mass_count: 0,
  freq_90pct_mass_count: 0,
  freq_90pct_mass_ratio: 1,
  freq_gini: 3,
}

const IMPL_ABBREV: Record<string, string> = {
  Transformers: 'hf-s',
  Tokenizers: 'hf-f',
  SentencePiece: 'sp',
  Tiktoken: 'tik',
  FastBPE: 'fbpe',
}

// Custom sorting function that matches the existing parseSortValue logic:
// numbers sort numerically (null/empty → Infinity), strings sort lexicographically
const numericSortFn: SortingFn<Record<string, unknown>> = (
  rowA: Row<Record<string, unknown>>,
  rowB: Row<Record<string, unknown>>,
  columnId: string,
) => {
  const va = rowA.getValue(columnId)
  const vb = rowB.getValue(columnId)
  const nullA = va == null || va === ''
  const nullB = vb == null || vb === ''
  if (nullA && nullB) return 0
  if (nullA) return 1
  if (nullB) return -1
  const na = parseFloat(String(va).replace(/,/g, ''))
  const nb = parseFloat(String(vb).replace(/,/g, ''))
  if (!isNaN(na) && !isNaN(nb)) return na - nb
  return String(va).toLowerCase().localeCompare(String(vb).toLowerCase())
}

const stringSortFn: SortingFn<Record<string, unknown>> = (
  rowA: Row<Record<string, unknown>>,
  rowB: Row<Record<string, unknown>>,
  columnId: string,
) => {
  const va = rowA.getValue(columnId)
  const vb = rowB.getValue(columnId)
  if (va == null && vb == null) return 0
  if (va == null) return 1
  if (vb == null) return -1
  return String(va).toLowerCase().localeCompare(String(vb).toLowerCase())
}

function renderCellValue(row: Record<string, unknown>, col: ColumnHeader) {
  const val = row[col.key]

  if (col.key === 'parity_en' && (val == null || val === '' || val === 0)) {
    return (
      <span
        className="cell-na"
        title="Parity is not applicable here — this language is the reference or has no parallel corpus."
      >
        —
      </span>
    )
  }

  if (val == null || val === '') return <span className="cell-string">—</span>

  if (col.key === 'model') {
    const name = String(row.model ?? '')
    const rev = row.revision && row.revision !== 'main' ? ` (rev: ${row.revision})` : ''
    const sub = row.subfolder ? ` (sub: ${row.subfolder})` : ''
    const tgt = row.target_mode ? ' (target)' : ''
    const text = name + rev + sub + tgt
    const url = `https://huggingface.co/${name}`
    return (
      <span className="cell-model">
        <a href={url} target="_blank" rel="noopener noreferrer" title={text}>
          {text}
        </a>
      </span>
    )
  }

  const numericVal = typeof val === 'number' ? val : Number(String(val).replace(/,/g, ''))
  const hasNumericVal = Number.isFinite(numericVal)
  const decimals = METRIC_DECIMALS[col.key] ?? DEFAULT_METRIC_DECIMALS

  if (PERCENTAGE_COLS.has(col.key) && hasNumericVal) {
    return <span className="cell-number">{(numericVal * 100).toFixed(decimals)}%</span>
  }

  if (col.key === 'tokenizer_algorithm') {
    const impl = row.tokenizer_implementation
      ? ` (${IMPL_ABBREV[row.tokenizer_implementation as string] ?? 'unk'})`
      : ''
    return <span className="cell-string">{String(val)}{impl}</span>
  }

  if (col.datatype === 'number' && hasNumericVal) {
    return <span className="cell-number">{numericVal.toFixed(decimals)}</span>
  }

  return <span className="cell-string">{String(val)}</span>
}

interface TooltipState {
  text: string
  x: number
  y: number
}

function ColTooltipPortal({ tooltip }: { tooltip: TooltipState | null }) {
  if (!tooltip) return null
  return createPortal(
    <div
      className="th-tooltip-portal"
      style={{ left: tooltip.x, top: tooltip.y }}
    >
      {tooltip.text}
    </div>,
    document.body,
  )
}

const ESTIMATED_ROW_HEIGHT = 36

export default function LeaderboardTable({
  data,
  columns: colHeaders,
  loading,
  isMultilingual = false,
  avgColumns = [],
  columnDescriptions = {},
}: Props) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'ebpb', desc: false },
  ])
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const tableContainerRef = useRef<HTMLDivElement>(null)

  const handleThMouseEnter = useCallback(
    (e: React.MouseEvent<HTMLTableCellElement>, desc: string) => {
      if (hideTimer.current) clearTimeout(hideTimer.current)
      const rect = e.currentTarget.getBoundingClientRect()
      setTooltip({ text: desc, x: rect.left + rect.width / 2, y: rect.bottom + 6 })
    },
    [],
  )

  const handleThMouseLeave = useCallback(() => {
    hideTimer.current = setTimeout(() => setTooltip(null), 120)
  }, [])

  // Build TanStack ColumnDef array from ColumnHeader props
  const columnDefs = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    return colHeaders.map((col) => {
      const isNumeric = col.datatype === 'number'
      return {
        id: col.key,
        accessorFn: (row) => {
          const v = row[col.key]
          if (v == null || v === '') return undefined
          return isNumeric ? parseFloat(String(v).replace(/,/g, '')) : v
        },
        sortingFn: isNumeric ? numericSortFn : stringSortFn,
        sortUndefined: 'last' as const,
        // header is rendered in JSX below (we need access to sorting state and tooltip)
        header: col.label,
        cell: ({ row }) => renderCellValue(row.original, col),
      }
    })
  }, [colHeaders])

  const table = useReactTable({
    data,
    columns: columnDefs,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    // Keep original row index stable for even/odd striping
    getRowId: (_row, index) => String(index),
  })

  const rows = table.getRowModel().rows

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => ESTIMATED_ROW_HEIGHT,
    overscan: 5,
  })

  const virtualItems = virtualizer.getVirtualItems()
  const totalSize = virtualizer.getTotalSize()

  // Padding rows to fill virtual space above and below rendered items
  const paddingTop = virtualItems.length > 0 ? (virtualItems[0]?.start ?? 0) : 0
  const paddingBottom =
    virtualItems.length > 0
      ? totalSize - (virtualItems[virtualItems.length - 1]?.end ?? 0)
      : 0

  const headerGroups = table.getHeaderGroups()

  return (
    <>
      <ColTooltipPortal tooltip={tooltip} />
      <div className="table-container" ref={tableContainerRef}>
        <table className="data-table">
          <thead>
            {headerGroups.map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const col = colHeaders.find((c) => c.key === header.id)
                  if (!col) return null
                  const isAvg = isMultilingual && avgColumns.includes(col.key)
                  const desc = columnDescriptions[col.key]
                  const sorted = header.column.getIsSorted()
                  return (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      onMouseEnter={desc ? (e) => handleThMouseEnter(e, desc) : undefined}
                      onMouseLeave={desc ? handleThMouseLeave : undefined}
                    >
                      <div className="th-content">
                        <div className="th-main">
                          {col.emoji && <span className="th-emoji">{col.emoji}</span>}
                          <span className="th-label">
                            {col.label}
                            {isAvg && <sup className="th-avg-sup">¹</sup>}
                          </span>
                          <span className={`th-sort${sorted ? ' th-sort--active' : ''}`}>
                            {sorted === 'asc' ? '▲' : sorted === 'desc' ? '▼' : '⇅'}
                          </span>
                        </div>
                        {col.unit && <span className="th-unit">{col.unit}</span>}
                      </div>
                    </th>
                  )
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={colHeaders.length} className="table-loading-cell">
                  <div className="table-loading-overlay">
                    <div className="loading-spinner" />
                    <span className="loading-text">Loading...</span>
                  </div>
                </td>
              </tr>
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan={colHeaders.length} className="table-empty">
                  No results match your filters.
                </td>
              </tr>
            ) : (
              <>
                {paddingTop > 0 && (
                  <tr style={{ height: paddingTop }}>
                    <td colSpan={colHeaders.length} />
                  </tr>
                )}
                {virtualItems.map((virtualRow) => {
                  const row = rows[virtualRow.index]
                  return (
                    <tr
                      key={row.id}
                      className={virtualRow.index % 2 === 0 ? 'row-odd' : 'row-even'}
                      style={{ height: virtualRow.size }}
                    >
                      {row.getVisibleCells().map((cell) => {
                        const col = colHeaders.find((c) => c.key === cell.column.id)
                        return (
                          <td
                            key={cell.id}
                            style={{
                              textAlign: col?.datatype === 'number' ? 'right' : 'left',
                            }}
                          >
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
                {paddingBottom > 0 && (
                  <tr style={{ height: paddingBottom }}>
                    <td colSpan={colHeaders.length} />
                  </tr>
                )}
              </>
            )}
          </tbody>
        </table>
      </div>
    </>
  )
}
