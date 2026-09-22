/**
 * Proveedores cuya URL lleva el identificador de cuenta dentro (Cloudflare:
 * `/client/v4/accounts/{account_id}/ai/v1`). La UI oculta la URL en los
 * proveedores de nube, asi que el Account ID se pide aparte y se inserta aqui.
 * Si se deja vacio, el marcador viaja tal cual y el backend lo rellena con la
 * variable de entorno del catalogo (CLOUDFLARE_ACCOUNT_ID).
 */
export const ACCOUNT_PLACEHOLDER = "{account_id}";

export function withAccount(template: string, accountId: string): string {
  return template.replace(ACCOUNT_PLACEHOLDER, accountId.trim() || ACCOUNT_PLACEHOLDER);
}

/** Recupera el Account ID de una URL ya resuelta ("" si aun lleva el marcador). */
export function accountFromUrl(template: string, url: string): string {
  const at = template.indexOf(ACCOUNT_PLACEHOLDER);
  if (at < 0 || !url || url.includes(ACCOUNT_PLACEHOLDER)) return "";
  const prefix = template.slice(0, at);
  const suffix = template.slice(at + ACCOUNT_PLACEHOLDER.length);
  if (!url.startsWith(prefix) || !url.endsWith(suffix)) return "";
  return url.slice(prefix.length, url.length - suffix.length);
}
