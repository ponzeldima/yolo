# Label Studio YOLO backend

This backend serves YOLO predictions at `POST /predict`, compatible with a Label
Studio ML backend. `start-ngrok.ps1` runs the backend with Waitress bound only
to `127.0.0.1` and creates an ngrok HTTPS tunnel. It does not expose its
internal port directly on the network.

## Prerequisites

1. Install ngrok for Windows and add it to `PATH`.
2. Create a free ngrok account and copy its authtoken.
3. Use the existing `fpv_detector\.venv` environment, or update the `$python`
   variable in `start-ngrok.ps1` to the Python executable that has PyTorch and
   can run the selected YOLO model.

## Start the backend

From the repository root, run:

```powershell
.\ml_backend\start-ngrok.ps1 `
  -ModelPath best.pt `
  -NgrokAuthtoken '2KkA2MAS4c5Ww0ZJe8KS4iZZVJ2_2NS5HimMghpuYPwDe2wE2' `
  -BackendPort 9091
```

Add these optional parameters only when a task's `image` value is a relative or
authenticated Label Studio URL, such as `/data/upload/...`:

```powershell
  -LabelStudioUrl 'https://your-label-studio.example.com' `
  -LabelStudioApiKey '<your-label-studio-api-token>'
```

The backend uses the API key only for requests to the configured Label Studio
origin. It is not written to disk or sent to external image hosts.

After the health check completes, ngrok displays a URL such as:

```text
Forwarding  https://abc123.ngrok-free.app -> http://localhost:9090
```

Use `https://abc123.ngrok-free.app` as the Label Studio ML backend URL and
click **Check Connection**. Keep this PowerShell window open while Label
Studio needs predictions. The free ngrok hostname changes each time it starts.
`BackendPort` defaults to `9090`; choose another free port, such as `9091`, if
Windows reserves it.

## Secure the public endpoint

The ngrok URL is public. Configure a Traffic Policy or OAuth protection in the
ngrok dashboard before using sensitive images or production data. Make sure
Label Studio can send any required authentication headers to the protected
endpoint. Do not publish the ngrok authtoken or Label Studio API token.

## Image inputs

The backend accepts:

- Label Studio local-file paths mounted beneath `DATA_ROOT` (default:
  `/app/data`);
- direct filesystem paths available to the backend process;
- HTTP(S) image URLs; and
- relative Label Studio paths when `LABEL_STUDIO_URL` is set.

Remote downloads are limited to 25 MB by default. Override this with
`-MaxImageMegabytes` when necessary.
