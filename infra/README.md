# Infra (minimal SAM skeleton)

This is a **starter** AWS SAM template — not a full production deploy.

## Intent

Show how the bridge could sit behind API Gateway HTTP API + Lambda.
The current FastAPI app is process-oriented; for Lambda you would typically:

1. Use [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter), or
2. Extract `process_scene` into a pure handler and map API Gateway events.

## Local build (optional)

```bash
# requires SAM CLI + Docker
sam build -t infra/template.yaml
# sam deploy --guided
```

Default `PIPELINE_MODE=mock` so the stack does not require Bedrock until you change it.
