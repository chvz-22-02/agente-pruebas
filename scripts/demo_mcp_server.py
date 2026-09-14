"""Servidor MCP de juguete para validar el banco de pruebas.

No forma parte de la solucion: sirve para comprobar que la UI, el agente, la
captura de JSON-RPC y MLflow funcionan antes de apuntar a un MCP real.

    python scripts/demo_mcp_server.py            # http://localhost:3333/mcp
    python scripts/demo_mcp_server.py --port 9000
"""

from __future__ import annotations

import argparse
import datetime as dt
import random

from mcp.server.mcpserver import MCPServer

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=3333)
parser.add_argument("--transport", default="streamable-http", choices=["streamable-http", "sse"])
args = parser.parse_args()

mcp = MCPServer("demo-pruebas", version="0.1.0")

_INVENTARIO = {
    "SKU-001": {"nombre": "Teclado mecanico", "stock": 42, "precio": 89.9},
    "SKU-002": {"nombre": "Monitor 27 pulgadas", "stock": 7, "precio": 249.0},
    "SKU-003": {"nombre": "Silla ergonomica", "stock": 0, "precio": 320.5},
}


@mcp.tool()
def hora_actual(zona: str = "local") -> str:
    """Devuelve la fecha y hora actuales del servidor."""
    return dt.datetime.now().isoformat(timespec="seconds") + f" ({zona})"


@mcp.tool()
def consultar_inventario(sku: str) -> dict:
    """Consulta un articulo del inventario por su SKU (p.ej. SKU-001)."""
    articulo = _INVENTARIO.get(sku.upper())
    if articulo is None:
        raise ValueError(f"SKU desconocido: {sku}. Disponibles: {', '.join(_INVENTARIO)}")
    return {"sku": sku.upper(), **articulo}


@mcp.tool()
def listar_inventario() -> list[dict]:
    """Lista todos los articulos del inventario."""
    return [{"sku": k, **v} for k, v in _INVENTARIO.items()]


@mcp.tool()
def calcular_pedido(sku: str, unidades: int) -> dict:
    """Calcula el importe de un pedido y comprueba si hay stock suficiente."""
    articulo = _INVENTARIO.get(sku.upper())
    if articulo is None:
        raise ValueError(f"SKU desconocido: {sku}")
    if unidades <= 0:
        raise ValueError("Las unidades deben ser un entero positivo")
    return {
        "sku": sku.upper(),
        "unidades": unidades,
        "importe": round(articulo["precio"] * unidades, 2),
        "stock_suficiente": articulo["stock"] >= unidades,
        "stock_disponible": articulo["stock"],
    }


@mcp.tool()
def herramienta_inestable(probabilidad_fallo: float = 0.5) -> str:
    """Falla de forma aleatoria. Util para probar como reacciona el agente a errores."""
    if random.random() < probabilidad_fallo:
        raise RuntimeError("Fallo simulado del servidor MCP")
    return "Todo correcto en esta ejecucion"


@mcp.resource("inventario://catalogo")
def catalogo() -> str:
    """Catalogo completo como recurso MCP."""
    lineas = [f"{k}: {v['nombre']} - {v['precio']} EUR (stock {v['stock']})" for k, v in _INVENTARIO.items()]
    return "\n".join(lineas)


if __name__ == "__main__":
    ruta = "/mcp" if args.transport == "streamable-http" else "/sse"
    print(f"Servidor MCP de prueba en http://{args.host}:{args.port}{ruta}  [{args.transport}]")
    mcp.run(transport=args.transport, host=args.host, port=args.port)
