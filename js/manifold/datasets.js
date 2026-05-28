export function parseCSV(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return [];
  const first = lines[0].split(',').map(s => s.trim());
  const headerIsNumeric = first.every(c => c.length > 0 && Number.isFinite(Number(c)));
  const dataLines = headerIsNumeric ? lines : lines.slice(1);
  const rows = [];
  for (const line of dataLines) {
    const parts = line.split(',').map(s => Number(s.trim()));
    if (parts.length < 2) continue;
    if (!parts.every(v => Number.isFinite(v))) continue;
    rows.push(parts);
  }
  if (rows.length === 0) return [];
  const widths = rows.map(r => r.length);
  const mode = widths.sort((a, b) => widths.filter(x => x === a).length - widths.filter(x => x === b).length).pop();
  return rows.filter(r => r.length === mode);
}
