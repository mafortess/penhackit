# parser_mapping.py

from penhackit.session.parser.parser_catalog import ACTION_PARSERS



def parse_command_result(action_name: str, result: dict) -> list[dict]:
    print(f"Parsing result for action: {action_name} with rc={result.get('rc')}...")

    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""
    parser_family = result.get("parser_family")

    parser = resolve_parser(
        action_name=action_name,
        parser_family=parser_family,
    )

    if parser is None:
        print(f"No parser found. action_name={action_name}, parser_family={parser_family}")
        print(f"Available parser keys sample: {list(ACTION_PARSERS.keys())[:30]}")

        return [{
            "type": "NO_EVENT",
            "action": action_name,
            "reason": "No parser found for this action",
        }]

    target_ip = result.get("target_ip") or result.get("target")
    target_port = result.get("target_port")

    try:
        return call_parser(
            parser=parser,
            stdout=stdout,
            stderr=stderr,
            target_ip=target_ip,
            target_port=target_port,
            result=result,
        )
    except Exception as exc:
        return [{
            "type": "TOOL_ERROR",
            "action": action_name,
            "source": "parser",
            "error": str(exc),
            "raw_hint": (stdout + "\n" + stderr)[-1000:],
        }]


def resolve_parser(action_name: str | None, parser_family: str | None = None):
    candidates = []

    if action_name:
        candidates.append(action_name)
        candidates.append(action_name.lower())
        candidates.append(action_name.upper())

    if parser_family:
        candidates.append(parser_family)
        candidates.append(parser_family.lower())
        candidates.append(parser_family.upper())

    for candidate in candidates:
        if candidate in ACTION_PARSERS:
            return ACTION_PARSERS[candidate]

    return None


def call_parser(
    parser,
    stdout: str,
    stderr: str,
    target_ip: str | None,
    target_port,
    result: dict,
) -> list[dict]:
    if target_port is not None:
        try:
            target_port = int(target_port)
        except Exception:
            pass

    try:
        return parser(stdout, stderr, target_ip, target_port)
    except TypeError:
        pass

    try:
        return parser(stdout, stderr, target_ip)
    except TypeError:
        pass

    try:
        return parser(stdout, target_ip)
    except TypeError:
        pass

    try:
        return parser(stdout)
    except TypeError as exc:
        return [{
            "type": "TOOL_ERROR",
            "source": "parser",
            "error": f"Parser signature not supported: {exc}",
            "raw_hint": (stdout + "\n" + stderr)[-1000:],
        }]
    


# def parse_command_result(action: dict, result: dict) -> list[dict]:
#     """
#     Entry point for converting a command execution result into normalized events.
    
#     action: action metadata dict from ACTIONS.
#     result: {
#         "cmd": str | None,
#         "rc": int,
#         "stdout": str,
#         "stderr": str,
#         "target_ip": optional,
#         "target_port": optional,
#         "target": optional,
#         ...
#     }
    
#     Return a list of events for updating the KB. Each event is a dict with a "type" field and other relevant data.
#     """
#     print(f"\nEVENTS:")
#     print("Building event from command result...")

#     if isinstance(action, dict):
#         action_name = action.get("name", "UNKNOWN_ACTION")
#     elif isinstance(action, str):
#         action_name = action
#     else:
#         action_name = "UNKNOWN_ACTION"

#     try:
#         rc = int(result.get("rc", 0))
#     except (ValueError, TypeError):
#         rc = -1

#     stdout = result.get("stdout", "") or ""
#     stderr = result.get("stderr", "") or ""

#     target_ip = result.get("target_ip")
#     target_port = result.get("target_port")
#     target = result.get("target")
#     target_domain = result.get("target_domain")

#     if rc is not None and int(rc) != 0:
#         return [{"type": "COMMAND_ERROR", 
#                  "action": action_name, 
#                  "rc": rc, 
#                  "stderr": (result.get("stderr", "") or "")[:500],
#                 }]

#     print(f"Parsing result for action: {action_name} with rc={rc}...")
#     parser = ACTION_PARSERS.get(action_name)

#     try:
#         if parser is None:
#             return [{
#                 "type": "NO_EVENT",
#                 "action": action_name,
#                 "reason": "No parser found for this action",
#             }]

#         try:
#             return parser(stdout, stderr, target_ip, target_port)

#         except Exception as e:
#             print(f"Error occurred while parsing result for action: {action_name}")
#             print(f"Error: {e}")
#             return [{
#                 "type": "PARSING_ERROR",
#                 "action": action_name,
#                 "reason": str(e),
#             }]
#     except Exception as e:
#         print(f"Unexpected error in parse_command_result for action: {action_name}")
#         print(f"Error: {e}")
#         return [{
#             "type": "PARSING_ERROR",
#             "action": action_name,
#             "reason": f"Unexpected error: {str(e)}",
#         }]
    
