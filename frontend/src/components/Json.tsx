type Props = { value: unknown; max?: number };

/** Bloque JSON plegable con recorte defensivo para payloads enormes. */
export default function Json({ value, max = 20000 }: Props) {
  let text: string;
  try {
    text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  } catch {
    text = String(value);
  }
  if (text && text.length > max) {
    text = `${text.slice(0, max)}\n... [truncado, ${text.length - max} caracteres]`;
  }
  return <pre>{text || "(vacio)"}</pre>;
}
