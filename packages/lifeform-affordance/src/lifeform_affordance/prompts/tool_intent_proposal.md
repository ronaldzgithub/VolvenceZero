You are deciding whether a lifeform should use one available tool during the
current conversation turn. Return only JSON that conforms to the schema.

Schema:
{schema}

Decision rules:
- Choose "invoke_tool" only when one descriptor clearly applies and all
  required parameters can be filled from the conversation context.
- Choose "final_answer" when the assistant should answer without a tool.
- Choose "pause" when the user asks to wait or hold the tool loop.
- Choose "stop" when the user asks to stop/cancel the current tool loop.
- Choose "continue" only when a previous tool result should be followed up
  without calling another tool.
- Never invent unavailable tool names.
- Never infer hidden state; use only the context below.

Context:
active_regime_id: {active_regime_id}
user_input: {user_input}
latest_response_text: {latest_response_text}
available_tools_json: {descriptors}
previous_steps_json: {previous_steps}
