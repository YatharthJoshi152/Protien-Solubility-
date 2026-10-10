import { useEffect, useMemo, useRef, useState } from 'react'

const AA = 'ACDEFGHIKLMNPQRSTVWY'.split('')
const MAIN_FIELDS = [
  ['MW_kDa', 'Molecular weight', 'kDa'],
  ['pI', 'Isoelectric point', 'pI'],
  ['Length', 'Sequence length', 'residues'],
  ['GRAVY', 'GRAVY', ''],
  ['Aromaticity', 'Aromaticity', '%'],
  ['Instability_Index', 'Instability index', ''],
  ['Aliphatic_Index', 'Aliphatic index', ''],
  ['Charged_pct', 'Percent charged', '%'],
  ['Hydrophobic_pct', 'Percent hydrophobic', '%'],
  ['Polar_pct', 'Percent polar', '%'],
  ['Acidic_pct', 'Percent acidic', '%'],
  ['Basic_pct', 'Percent basic', '%'],
] as const

type Range = { min: number; max: number }
type Meta = {
  categories: Record<string, string[]>
  training_ranges: Record<string, Range>
  parameter_descriptions: Record<string, string>
  model_card: Record<string, any>
  example_protein?: { name: string; identifier: string; sequence: string; cell_location: string; gene_type: string; parameters: Record<string, number> }
}
type Result = {
  name: string
  identifier?: string | null
  final_score: number
  band: string
  per_model: { lr: number; svm: number; rf: number }
  spread: number
  agreement: string
  outlier_model?: string | null
  lr_contributions: { feature: string; direction: string; size: number }[]
  rf_importances: { feature: string; importance: number }[]
  warnings: string[]
  parameters: Record<string, number>
}
type FormState = {
  name: string
  identifier: string
  sequence: string
  cellLocation: string
  geneType: string
  parameters: Record<string, number>
  editedFields: string[]
}

const emptyForm: FormState = { name: '', identifier: '', sequence: '', cellLocation: '', geneType: '', parameters: {}, editedFields: [] }

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed.')
  return data as T
}

function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [page, setPage] = useState(window.location.pathname)
  const [metaError, setMetaError] = useState('')

  useEffect(() => {
    api<Meta>('/meta').then(setMeta).catch((error: Error) => setMetaError(error.message))
    const handlePop = () => setPage(window.location.pathname)
    window.addEventListener('popstate', handlePop)
    return () => window.removeEventListener('popstate', handlePop)
  }, [])

  const navigate = (event: React.MouseEvent<HTMLAnchorElement>, path: string) => {
    event.preventDefault()
    window.history.pushState({}, '', path)
    setPage(path)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="site-shell">
      <header className="site-header">
        <div className="wordmark" aria-label="E. coli Protein Solubility Predictor">
          <span className="instrument-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>E. coli Protein Solubility Predictor</span>
        </div>
        <nav aria-label="Primary navigation">
          <a className={page === '/' ? 'active' : ''} href="/" onClick={(event) => navigate(event, '/')}>Predictor</a>
          <a className={page === '/batch' ? 'active' : ''} href="/batch" onClick={(event) => navigate(event, '/batch')}>Batch</a>
          <a className={page === '/how-this-works' ? 'active' : ''} href="/how-this-works" onClick={(event) => navigate(event, '/how-this-works')}>How this works</a>
        </nav>
      </header>
      <main>
        {metaError && <div className="global-error" role="alert">{metaError}</div>}
        {page === '/batch' ? <BatchPage /> : page === '/how-this-works' ? <HowPage meta={meta} /> : <PredictorPage meta={meta} />}
      </main>
      <footer><span>Screening instrument for E. coli proteins.</span><span>Score is not a lab guarantee.</span></footer>
    </div>
  )
}

function PredictorPage({ meta }: { meta: Meta | null }) {
  const [form, setForm] = useState<FormState>(emptyForm)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [sequenceLocked, setSequenceLocked] = useState(false)
  const [busy, setBusy] = useState(false)
  const [whatIf, setWhatIf] = useState({ pI: 7, MW_kDa: 50, Hydrophobic_pct: 40 })
  const debounceRef = useRef<number | null>(null)
  const baseRequestRef = useRef<FormState | null>(null)

  const setField = (field: keyof FormState, value: string) => setForm((current) => ({ ...current, [field]: value }))
  const setParameter = (field: string, value: string) => {
    const number = value === '' ? undefined : Number(value)
    setForm((current) => ({
      ...current,
      parameters: { ...current.parameters, ...(number === undefined || Number.isNaN(number) ? {} : { [field]: number }) },
      editedFields: current.sequence && sequenceLocked && !current.editedFields.includes(field) ? [...current.editedFields, field] : current.editedFields,
    }))
  }

  const calculateFeatures = async () => {
    setError('')
    if (!form.sequence.trim()) {
      setError('Paste an amino acid sequence before calculating parameters.')
      return
    }
    setBusy(true)
    try {
      const data = await api<{ sequence: string; features: Record<string, number> }>('/features', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sequence: form.sequence }) })
      setForm((current) => ({ ...current, sequence: data.sequence, parameters: { ...current.parameters, ...data.features, MW_kDa: data.features.MW_kDa_calc, pI: data.features.pI_calc }, editedFields: [] }))
      setSequenceLocked(true)
      setNotice('Parameters calculated from the sequence. Derived fields are locked until you choose Edit manually.')
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const loadExample = async () => {
    setError('')
    try {
      const data = meta?.example_protein ?? (await api<Meta>('/meta')).example_protein
      if (!data) throw new Error('The backend did not provide an example protein.')
      setForm({ name: data.name, identifier: data.identifier, sequence: data.sequence, cellLocation: data.cell_location, geneType: data.gene_type, parameters: data.parameters, editedFields: [] })
      setSequenceLocked(true)
      setResult(null)
      setNotice('Example protein loaded from the backend.')
    } catch (requestError) {
      setError((requestError as Error).message)
    }
  }

  const reset = () => {
    setForm(emptyForm)
    setResult(null)
    setError('')
    setNotice('')
    setSequenceLocked(false)
    setWhatIf({ pI: 7, MW_kDa: 50, Hydrophobic_pct: 40 })
    baseRequestRef.current = null
  }

  const predict = async (override?: Partial<FormState['parameters']>, keepBase = false) => {
    setError('')
    const parameters = { ...form.parameters, ...Object.fromEntries(Object.entries(override ?? {}).filter(([, value]) => value !== undefined)) } as Record<string, number>
    const required = ['MW_kDa', 'pI', 'Length', 'GRAVY', 'Instability_Index', 'Aliphatic_Index', 'Hydrophobic_pct', 'Charged_pct', ...AA.map((aa) => `${aa}_pct`)]
    const missing = required.filter((field) => parameters[field] === undefined)
    if (!form.sequence && missing.length) {
      setError(`Enter all required parameters or calculate them from a sequence. Missing: ${missing.join(', ')}.`)
      return
    }
    setBusy(true)
    try {
      const data = await api<Result>('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: form.name || 'Unnamed protein', identifier: form.identifier || null, sequence: form.sequence || null, parameters, cell_location: form.cellLocation || null, gene_type: form.geneType || null, edited_fields: form.editedFields }) })
      setResult(data)
      if (!keepBase) {
        baseRequestRef.current = { ...form, parameters: { ...parameters } }
        setWhatIf({ pI: Number(parameters.pI ?? 7), MW_kDa: Number(parameters.MW_kDa ?? 50), Hydrophobic_pct: Number(parameters.Hydrophobic_pct ?? 40) })
      }
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const runWhatIf = (field: 'pI' | 'MW_kDa' | 'Hydrophobic_pct', value: number) => {
    const next = { ...whatIf, [field]: value }
    setWhatIf(next)
    if (debounceRef.current) window.clearTimeout(debounceRef.current)
    debounceRef.current = window.setTimeout(() => predict(next, true), 250)
  }

  const parameterDescriptions = meta?.parameter_descriptions ?? {}
  const ranges = meta?.training_ranges ?? {}

  return (
    <>
      <section className="page-intro">
        <p className="eyebrow">Prediction console</p>
        <h1>Estimate protein solubility in E. coli</h1>
        <p>Enter a sequence or physicochemical parameters. Three trained models return probabilities and a combined score.</p>
      </section>
      <div className="instrument-grid">
        <section className="panel form-panel" aria-labelledby="input-title">
          <div className="panel-heading"><div><p className="eyebrow">Input</p><h2 id="input-title">Protein details</h2></div><span className="step-index">01</span></div>
          <div className="field-grid two-up">
            <label><span>Protein name <small>optional label</small></span><input value={form.name} onChange={(event) => setField('name', event.target.value)} /></label>
            <label><span>Identifier <small>optional, for example a B number</small></span><input value={form.identifier} onChange={(event) => setField('identifier', event.target.value)} /></label>
          </div>
          <label className="sequence-label"><span>Amino acid sequence <small>standard amino acids only, minimum 10 residues</small></span><textarea className="sequence-input" value={form.sequence} onChange={(event) => { setField('sequence', event.target.value); setSequenceLocked(false); setNotice('') }} placeholder="Paste a protein sequence" spellCheck={false} /></label>
          <div className="action-row"><button className="primary-button" onClick={calculateFeatures} disabled={busy}>Calculate parameters from sequence</button>{sequenceLocked && <button className="text-button" onClick={() => { setSequenceLocked(false); setNotice('Derived fields can now be edited. Edited fields will be marked.') }}>Edit manually</button>}</div>
          {notice && <p className="notice" role="status">{notice}</p>}
          <div className="field-grid two-up category-grid">
            <label><span>Cell location <small>from training data</small></span><select value={form.cellLocation} onChange={(event) => setField('cellLocation', event.target.value)}><option value="">Select a category</option>{(meta?.categories?.Cell_location ?? []).map((option) => <option key={option}>{option}</option>)}</select></label>
            <label><span>Type of gene product <small>from training data</small></span><select value={form.geneType} onChange={(event) => setField('geneType', event.target.value)}><option value="">Select a category</option>{(meta?.categories?.Gene_type ?? []).map((option) => <option key={option}>{option}</option>)}</select></label>
          </div>
          <div className="section-rule"><span>Parameters</span><span className="rule-note">Values outside training range warn, but do not block.</span></div>
          <div className="field-grid two-up">
            {MAIN_FIELDS.map(([field, label, unit]) => <ParameterField key={field} field={field} label={label} unit={unit} value={form.parameters[field]} range={ranges[field]} description={parameterDescriptions[field]} locked={sequenceLocked} edited={form.editedFields.includes(field)} onChange={setParameter} />)}
          </div>
          <details className="composition-details">
            <summary><span>Amino acid composition</span><span className="composition-total">Total {AA.reduce((sum, aa) => sum + (form.parameters[`${aa}_pct`] ?? 0), 0).toFixed(2)}%</span></summary>
            <p className="helper-text">Percentages should total 100.00 within 0.01.</p>
            <div className="field-grid three-up">{AA.map((aa) => <ParameterField key={aa} field={`${aa}_pct`} label={`${aa} percent`} unit="%" value={form.parameters[`${aa}_pct`]} range={ranges[`${aa}_pct`]} description="Percent of sequence length." locked={sequenceLocked} edited={form.editedFields.includes(`${aa}_pct`)} onChange={setParameter} compact />)}</div>
          </details>
          {error && <p className="error-message" role="alert">{error}</p>}
          <div className="form-actions"><button className="primary-button" onClick={() => predict()} disabled={busy}>{busy ? 'Scoring...' : 'Predict'}</button><button className="secondary-button" onClick={reset}>Reset</button><button className="secondary-button" onClick={loadExample}>Load example protein</button></div>
          <p className="fine-print">The score is the predicted probability that the protein is soluble when expressed in E. coli. Soluble means measured solubility of at least 50 percent in the training data. It is not a predicted solubility percentage or a lab guarantee.</p>
        </section>
        <ResultPanel result={result} whatIf={whatIf} onWhatIf={runWhatIf} />
      </div>
    </>
  )
}

function ParameterField({ field, label, unit, value, range, description, locked, edited, onChange, compact = false }: { field: string; label: string; unit: string; value?: number; range?: Range; description?: string; locked: boolean; edited: boolean; onChange: (field: string, value: string) => void; compact?: boolean }) {
  const outOfRange = value !== undefined && range && (value < range.min || value > range.max)
  return <label className={`parameter-field ${compact ? 'compact' : ''}`}><span>{label} {unit && <small>{unit}</small>} {edited && <em>edited</em>}</span><div className="number-wrap"><input type="number" step="any" value={value ?? ''} disabled={locked} onChange={(event) => onChange(field, event.target.value)} aria-describedby={`${field}-note`} />{outOfRange && <b className="range-flag">outside range</b>}</div><small id={`${field}-note`} className="field-note">{description}{range ? ` Training range: ${range.min.toPrecision(3)} to ${range.max.toPrecision(3)}.` : ''}</small></label>
}

function ResultPanel({ result, whatIf, onWhatIf }: { result: Result | null; whatIf: { pI: number; MW_kDa: number; Hydrophobic_pct: number }; onWhatIf: (field: 'pI' | 'MW_kDa' | 'Hydrophobic_pct', value: number) => void }) {
  const exportJson = () => {
    if (!result) return
    navigator.clipboard?.writeText(JSON.stringify(result, null, 2))
  }
  const downloadCsv = () => {
    if (!result) return
    const content = ['name,final_score,band,lr_probability,svm_probability,rf_probability,spread,agreement,outlier_model', `${JSON.stringify(result.name)},${result.final_score},${JSON.stringify(result.band)},${result.per_model.lr},${result.per_model.svm},${result.per_model.rf},${result.spread},${JSON.stringify(result.agreement)},${JSON.stringify(result.outlier_model ?? '')}`].join('\n')
    const link = document.createElement('a')
    link.href = URL.createObjectURL(new Blob([content], { type: 'text/csv' }))
    link.download = 'solubility-result.csv'
    link.click()
    URL.revokeObjectURL(link.href)
  }
  return <section className="panel result-panel" aria-labelledby="result-title"><div className="panel-heading"><div><p className="eyebrow">Output</p><h2 id="result-title">Prediction result</h2></div><span className="step-index">02</span></div>{!result ? <div className="empty-result"><span className="empty-mark" aria-hidden="true">+</span><h3>Awaiting input</h3><p>Submit a sequence or a complete parameter set to see the three model probabilities, their agreement, and the combined score.</p></div> : <><div className="score-readout"><div><span className="eyebrow">Final score</span><strong className="score-number">{result.final_score.toFixed(2)}</strong></div><span className="band-label">{result.band}</span></div><ScoreScale score={result.final_score} /><p className="score-definition">Predicted probability of measured solubility at least 50 percent for this E. coli protein. The score is not a predicted solubility percentage.</p><div className="model-table" role="table" aria-label="Model probabilities"><div className="model-row model-header" role="row"><span>Model</span><span>Probability</span><span>Scale</span></div>{[['lr', 'Logistic regression'], ['svm', 'Calibrated SVM'], ['rf', 'Random forest']].map(([key, label]) => <div className="model-row" role="row" key={key}><span>{label}</span><code>{result.per_model[key as keyof Result['per_model']].toFixed(4)}</code><span className="mini-bar"><i style={{ width: `${result.per_model[key as keyof Result['per_model']] * 100}%` }} /></span></div>)}</div><div className="agreement-line"><strong>{result.agreement}</strong><span>Spread {result.spread.toFixed(4)}</span>{result.outlier_model && <span>Outlier: {result.outlier_model.toUpperCase()}</span>}</div><section className="result-section"><div className="section-rule"><span>What if</span></div><p className="helper-text">Move one slider to rerun the prediction after a short delay. The form values do not change.</p><Slider label="pI" value={whatIf.pI} min={0} max={14} step={0.01} onChange={(value) => onWhatIf('pI', value)} /><Slider label="Molecular weight, kDa" value={whatIf.MW_kDa} min={1} max={200} step={0.1} onChange={(value) => onWhatIf('MW_kDa', value)} /><Slider label="Percent hydrophobic" value={whatIf.Hydrophobic_pct} min={0} max={100} step={0.1} onChange={(value) => onWhatIf('Hydrophobic_pct', value)} /></section>{result.warnings.length > 0 && <section className="warning-box" aria-label="Warnings"><h3>Warnings</h3><ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></section>}<div className="result-actions"><button className="secondary-button" onClick={exportJson}>Copy result as JSON</button><button className="secondary-button" onClick={downloadCsv}>Download result as CSV</button><button className="secondary-button" onClick={() => window.print()}>Print</button></div></>}</section>
}

function ScoreScale({ score }: { score: number }) { return <div className="score-scale"><div className="scale-bands"><span className="insoluble">Likely insoluble</span><span className="uncertain">Uncertain</span><span className="soluble">Likely soluble</span><i className="score-marker" style={{ left: `${score * 100}%` }} /></div><div className="scale-ticks"><span>0.0</span><span>0.5</span><span>1.0</span></div></div> }
function Slider({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) { return <label className="slider-field"><span>{label}<code>{value.toFixed(2)}</code></span><input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} /></label> }

function BatchPage() {
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [downloadUrl, setDownloadUrl] = useState('')
  const runBatch = async () => { if (!file) { setError('Choose a CSV file first.'); return } setError(''); setStatus('Running batch prediction...'); setDownloadUrl(''); const formData = new FormData(); formData.append('file', file); try { const response = await fetch('/predict_batch', { method: 'POST', body: formData }); const data = await response.text(); if (!response.ok) throw new Error(data || 'The batch request failed.'); setDownloadUrl(URL.createObjectURL(new Blob([data], { type: 'text/csv' }))); setStatus('Batch prediction complete.'); } catch (requestError) { setError((requestError as Error).message); setStatus('') } }
  return <><section className="page-intro"><p className="eyebrow">Batch mode</p><h1>Score a CSV of proteins</h1><p>Upload one row per protein. The backend returns the three model probabilities, final score, agreement label, spread, and outlier model.</p></section><section className="single-column"><div className="panel batch-panel"><div className="panel-heading"><div><p className="eyebrow">CSV input</p><h2>Upload batch</h2></div><span className="step-index">01</span></div><label className="file-drop"><span>CSV file</span><input type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /><small>{file ? file.name : 'Select a UTF-8 CSV file.'}</small></label><button className="primary-button" onClick={runBatch}>Run batch prediction</button>{status && <p className="notice" role="status">{status}</p>}{error && <p className="error-message" role="alert">{error}</p>}{downloadUrl && <a className="download-link" href={downloadUrl} download="solubility_predictions.csv">Download predictions CSV</a>}<div className="section-rule"><span>Columns</span></div><p>Include `Sequence` for sequence-based inputs. You can also include numeric feature columns and these category columns. The output includes one row per input row.</p><div className="column-list"><code>name</code><code>identifier</code><code>Sequence</code><code>Cell_location</code><code>Gene_type</code><code>MW_kDa</code><code>pI</code><code>Length</code><code>GRAVY</code><code>Instability_Index</code><code>Aliphatic_Index</code><code>Hydrophobic_pct</code><code>Charged_pct</code><code>A_pct through Y_pct</code></div><p className="fine-print">The supplied sample file is in `data/sample_batch.csv`.</p></div></section></>
}

function HowPage({ meta }: { meta: Meta | null }) {
  const card = meta?.model_card ?? {}
  const metrics = card.metrics ?? {}
  return <><section className="page-intro"><p className="eyebrow">Model notes</p><h1>How this works</h1><p>This page describes the data, features, models, and limits of the predictor. Metrics are read from the saved model card.</p></section><section className="single-column"><article className="panel prose-panel"><InfoSection title="Data"><p>The training table contains E. coli K-12 proteins labeled by measured solubility. A label of soluble means the measured solubility is at least 50 percent. The training workflow groups rows by B number when splitting, so related constructs stay together as far as the available identifiers allow.</p></InfoSection><InfoSection title="Features"><p>The shared feature function validates standard amino acids, then calculates composition percentages, sequence length, GRAVY, aromaticity, instability, aliphatic index, hydrophobic, charged, polar, acidic, and basic percentages. Molecular weight and pI for a new sequence are calculated with Biopython. The aliphatic index includes alanine, valine, isoleucine, and leucine.</p></InfoSection><InfoSection title="Three model views"><p>Logistic regression uses an extended numeric feature set, spline terms for log molecular weight and pI, and one-hot category fields. The SVM uses all 27 numeric fields, median imputation, scaling, and an RBF kernel with calibrated probabilities. The random forest uses 300 trees, balanced classes, square-root feature sampling, and the same 27 numeric fields.</p></InfoSection><InfoSection title="Combining the models"><p>The models first produce three probabilities. A small logistic regression stacker is trained only on out-of-fold predictions from grouped cross-validation. It is used only when its grouped cross-validation ROC-AUC beats every base model. Otherwise the final score is the simple average. The interface reports the spread between the three model probabilities.</p></InfoSection><InfoSection title="Model performance"><MetricTable title="Held-out test" data={metrics.held_out_test} /><MetricTable title="Grouped cross-validation" data={metrics.grouped_cross_validation} /></InfoSection><InfoSection title="Limitations"><ul className="plain-list">{(card.limitations ?? ['Not available']).map((item: string) => <li key={item}>{item}</li>)}</ul><p>Scores are screening outputs. They do not replace a laboratory measurement.</p></InfoSection><InfoSection title="Saved model versions"><ul className="plain-list">{Object.entries(card.library_versions ?? {}).map(([key, value]) => <li key={key}><span>{key}</span><code>{String(value)}</code></li>)}</ul></InfoSection></article></section></>
}

function InfoSection({ title, children }: { title: string; children: React.ReactNode }) { return <section className="info-section"><div className="section-rule"><span>{title}</span></div>{children}</section> }
function MetricTable({ title, data }: { title: string; data?: Record<string, Record<string, number>> }) { const rows = ['lr', 'svm', 'rf', 'ensemble']; const metricKeys = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']; return <div className="metric-block"><h3>{title}</h3><div className="metric-table" role="table"><div className="metric-row metric-header"><span>Model</span>{metricKeys.map((key) => <span key={key}>{key.replace('_', ' ')}</span>)}</div>{rows.map((row) => <div className="metric-row" key={row}><span>{row === 'lr' ? 'Logistic regression' : row === 'svm' ? 'Calibrated SVM' : row === 'rf' ? 'Random forest' : 'Combined score'}</span>{metricKeys.map((key) => <span key={key}>{data?.[row]?.[key] !== undefined ? Number(data[row][key]).toFixed(3) : 'Not available'}</span>)}</div>)}</div></div> }

export default App
