// Python's json.dumps for the decision workers: ", " and ": " separators, optional ensure_ascii escaping;
// with indent, one item a line and "," at line ends, as json.dumps(..., indent=n) writes it.
export function pyJsonDumps(value, ensureAscii, indent = null) {
  const hex = (code) => "\\u" + code.toString(16).padStart(4, "0");
  const SHORT = { '"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f" };
  const str = (s) => {
    let out = '"';
    for (const ch of s) {
      const code = ch.codePointAt(0);
      if (ch in SHORT) out += SHORT[ch];
      else if (code < 0x20) out += hex(code);
      else if (ensureAscii && code > 0xffff) {
        const u = code - 0x10000;
        out += hex(0xd800 + (u >> 10)) + hex(0xdc00 + (u & 0x3ff));
      } else if (ensureAscii && code > 0x7f) out += hex(code);
      else out += ch;
    }
    return out + '"';
  };
  const block = (open, close, items, depth) => {
    if (!items.length) return open + close;
    if (indent === null) return open + items.join(", ") + close;
    const pad = " ".repeat(indent * (depth + 1));
    return `${open}\n${pad}${items.join(`,\n${pad}`)}\n${" ".repeat(indent * depth)}${close}`;
  };
  const go = (x, depth) => {
    if (x === null || x === undefined) return "null";
    if (typeof x === "string") return str(x);
    if (typeof x === "boolean") return x ? "true" : "false";
    if (typeof x === "number") return String(x);
    if (Array.isArray(x)) return block("[", "]", x.map((v) => go(v, depth + 1)), depth);
    return block("{", "}", Object.entries(x).map(([k, v]) => `${str(k)}: ${go(v, depth + 1)}`), depth);
  };
  return go(value, 0);
}
