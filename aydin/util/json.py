import json
import jsonpickle


def encode_indent(object):

    return json.dumps(json.loads(jsonpickle.encode(object)), indent=4, sort_keys=True)
