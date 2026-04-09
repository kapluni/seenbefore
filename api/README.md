---
title: I've Seen This Before API
emoji: "\U0001F50D"
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# I've Seen This Before - API

Semantic similarity API that maps Soviet anti-Zionist propaganda to modern antisemitic rhetoric.

## Endpoints

- `POST /api/analyze` - Analyze text for Soviet propaganda echoes
- `GET /api/health` - Health check

## Usage

```bash
curl -X POST https://kapluni-ive-seen-this-before-api.hf.space/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Zionism is racism", "top_k": 5}'
```
