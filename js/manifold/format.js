function pad(s, width) {
  return s.length >= width ? s : ' '.repeat(width - s.length) + s;
}

export function formatVec3(v, { digits = 3, width = 6 } = {}) {
  const parts = Array.from(v).map(x => pad(Number(x).toFixed(digits), width));
  return '(' + parts.join(', ') + ')';
}

export function formatMatrix(M, { digits = 3, maxRows = null, maxCols = null, width = 7 } = {}) {
  const rows = maxRows !== null ? M.slice(0, maxRows) : M;
  const truncatedRows = maxRows !== null && M.length > maxRows;
  const lines = rows.map(row => {
    const arr = Array.from(row);
    const truncatedCols = maxCols !== null && arr.length > maxCols;
    const shown = maxCols !== null ? arr.slice(0, maxCols) : arr;
    const cells = shown.map(x => pad(Number(x).toFixed(digits), width));
    return '[ ' + cells.join(', ') + (truncatedCols ? ', ...' : '') + ' ]';
  });
  if (truncatedRows) lines.push('  ...');
  return lines.join('\n');
}

export function formatTable(headers, rows) {
  const widths = headers.map((h, i) => {
    let w = String(h).length;
    for (const row of rows) w = Math.max(w, String(row[i]).length);
    return w;
  });
  const fmtRow = (cells) => cells.map((c, i) => pad(String(c), widths[i])).join(' | ');
  const headerLine = fmtRow(headers);
  return [headerLine, ...rows.map(fmtRow)].join('\n');
}
