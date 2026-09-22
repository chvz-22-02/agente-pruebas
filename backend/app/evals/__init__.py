"""Evaluacion de agentes con usuarios simulados y un evaluador LLM.

Flujo de un caso de evaluacion:

    persona (personas.json) + consulta (consultas.json)
        -> simulador: escribe como la persona para alcanzar el objetivo
        -> agente bajo prueba: el mismo bucle que usa el chat, con el MCP real
        -> ... N turnos, hasta que la persona da la conversacion por cerrada
        -> verificaciones deterministas sobre las cifras del valor esperado
        -> evaluador LLM: juzga cada criterio de la rubrica con la transcripcion
        -> puntuacion agregada por codigo (pesos y criterios obligatorios)

Todo queda en SQLite y en MLflow, reutilizando la jerarquia del chat: una
ejecucion de evaluacion es una sesion y cada consulta una conversacion.
"""
