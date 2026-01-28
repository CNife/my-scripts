# 草案：雪球用户发帖爬虫

## Requirements (confirmed)
- 爬取雪球用户某个时间段内的发帖
- 示例用户主页：https://xueqiu.com/u/9493911686
- 输出格式：Markdown
- 使用项目现有的 Python 3.14 + uv 依赖管理

## Technical Decisions
- **API 方案**: 研究雪球网移动端或网页端 API，获取用户发帖列表
- **Cookie 管理**: 需要手动获取并配置雪球网的认证 Cookie
- **数据解析**: 使用 requests 库进行 HTTP 请求，json 解析响应
- **Markdown 生成**: 自定义函数将帖子内容转换为 Markdown 格式
- **时间范围过滤**: 支持按时间段筛选帖子
- **输出**: 保存为单个 Markdown 文件，包含帖子标题、内容、时间、阅读量等信息

## Research Findings
- **API 结构**: 雪球网使用前后端分离架构，核心数据通过 AJAX 接口获取
- **认证方式**: 需要携带 Cookie 进行身份验证（主要是 xq_a_token 和 u 字段）
- **反爬虫**: 可能有频率限制和请求头检查
- **pysnowball 库**: 存在第三方库，但主要用于股票数据，可能需要扩展或直接调用 API

## Open Questions
- 是否需要支持登录功能，还是手动配置 Cookie？
- 需要爬取哪些具体字段（标题、内容、时间、评论数、转发数、阅读数等）？
- 是否需要处理图片、视频等媒体内容？
- 输出的 Markdown 格式需要包含哪些元素？

## Scope Boundaries
- **INCLUDE**:
  - 用户发帖列表的获取和解析
  - 按时间段筛选帖子
  - 转换为 Markdown 格式
  - 保存为本地文件
- **EXCLUDE**:
  - 自动登录功能
  - 图片/视频的下载和保存
  - 评论和转发内容的爬取
