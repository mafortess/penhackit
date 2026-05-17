## Software modules

| Module | Responsibility | Main input | Main output |
|---|---|---|---|
| CLI | Provides the user interface and routes each option to the corresponding service. | User options | Service invocation |
| Session | Creates, configures and executes agent sessions. | Session configuration, mode, target | KB, logs, transitions |
| Training | Builds datasets and trains decision models. | Recorded sessions, datasets | Trained model, metrics |
| Report | Generates technical reports from session results. | Final session KB | Markdown/PDF report |
| Settings | Loads and manages persistent configuration. | `settings.json` | Runtime configuration |
| Environment | Detects system capabilities and available tools. | Operating system, installed tools | Environment profile |