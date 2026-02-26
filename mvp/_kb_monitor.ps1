$kb = "<ABS_PATH_TO_KB_JSON>"
while ($true) {
  cls
  if (Test-Path $kb) { gc $kb -Raw } else { "Waiting for kb.json" }
  sleep -m 500
}