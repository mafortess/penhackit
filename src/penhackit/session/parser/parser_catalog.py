from penhackit.session.parser.control_parser import CONTROL_PARSERS
from penhackit.session.parser.recon_parser import RECON_PARSERS
from penhackit.session.parser.local_context_parser import LOCAL_CONTEXT_PARSERS
from penhackit.session.parser.enumeration_parser import ENUMERATION_PARSERS
from penhackit.session.parser.vulnerability_parser import VULNERABILITY_PARSERS
from penhackit.session.parser.credential_parser import CREDENTIALS_PARSERS
from penhackit.session.parser.exploitation_parser import EXPLOITATION_PARSERS
from penhackit.session.parser.post_exploitation_parser import POST_EXPLOIT_PARSERS

ACTION_PARSERS = {
    **CONTROL_PARSERS,
    **LOCAL_CONTEXT_PARSERS,
    **RECON_PARSERS,
    **ENUMERATION_PARSERS,
    **VULNERABILITY_PARSERS,
    **CREDENTIALS_PARSERS,
    **EXPLOITATION_PARSERS,
    **POST_EXPLOIT_PARSERS,
}