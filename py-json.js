// Python's json.dumps for the decision workers: ", " and ": " separators, optional ensure_ascii escaping.
export function pyJsonDumps(value, ensureAscii) {
  const hex = (code) => "\\u" + code.toString(16).padStart(4, "0");
  const str = (s) => {
    let out = '"';
    for (const ch of s) {
      const code = ch.codePointAt(0);
      if (ch === '"') out += '\\"';
      else if (ch === "\\") out += "\\\\";
      else if (ch === "\n") out += "\\n";
      else if (ch === "\r") out += "\\r";
      else if (ch === "\t") out += "\\t";
      else if (code < 0x20) out += hex(code);
      else if (ensureAscii && code > 0xffff) {
        const u = code - 0x10000;
        out += hex(0xd800 + (u >> 10)) + hex(0xdc00 + (u & 0x3ff));
      } else if (ensureAscii && code > 0x7f) out += hex(code);
      else out += ch;
    }
    return out + '"';
  };
  const go = (x) => {
    if (x === null || x === undefined) return "null";
    if (typeof x === "string") return str(x);
    if (typeof x === "boolean") return x ? "true" : "false";
    if (typeof x === "number") return String(x);
    if (Array.isArray(x)) return "[" + x.map(go).join(", ") + "]";
    return "{" + Object.entries(x).map(([k, v]) => `${str(k)}: ${go(v)}`).join(", ") + "}";
  };
  return go(value);
}
