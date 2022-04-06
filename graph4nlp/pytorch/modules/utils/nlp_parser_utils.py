def get_stanza_properties(properties_args):
    """
    Return properties for stanza from omega conf
    """
    ret = {}

    def convert_bool2string(var: bool):
        return "false" if var is False else "true"

    for k, v in properties_args.items():
        if isinstance(v, dict):
            v_string = ""
            for kk, vv in v.items():
                if not isinstance(vv, bool):
                    raise RuntimeError("The properties args for stanza is incorrect.")
                v_string += str(kk) + "=" + convert_bool2string(vv) + ","
            v_string = v_string[:-1]
            ret[str(k)] = v_string
        elif isinstance(v, bool):
            ret[str(k)] = v
        elif isinstance(v, list):
            ret[str(k)] = ",".join(v)
        else:
            raise RuntimeError("The properties args for stanza is incorrect.")
    return ret
