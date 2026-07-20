请渲染下面的合成轨迹文本槽。

`trajectory_id`、`scenario_ref`、`slots` 与 `output_schema` 都是只读协议字段。仅返回：

`{"trajectory_id":"原值","slots":[{"turn_id":"原值","text":"渲染文本"}, ...]}`

输入：
{{REQUEST_JSON}}
