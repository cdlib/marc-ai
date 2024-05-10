from collections import OrderedDict

import pymarc


def get_record_values(record, location):
    split = location.split("$")

    if len(split) == 1:
        tag = split[0]
        code = None
    elif len(split) == 2:
        tag, code = split
    else:
        raise ValueError("Invalid location")

    # Find fields matching tag
    fields = record.get_fields(tag)

    results = []
    for current_value in fields:
        if current_value is not None:
            if code is not None:
                values = current_value.get_subfields(code)
                results.extend(values)
            elif isinstance(current_value, pymarc.Field):
                results.append(current_value.value())

    return " ".join(results)


def record_dict(record):
    d = OrderedDict()

    d["cid"] = get_record_values(record, "CID")
    d["id"] = get_record_values(record, "001")

    fixed_data = get_record_values(record, "008")
    d["pub_date"] = fixed_data[7:11]
    d["pub_place"] = fixed_data[15:18]
    d["language"] = fixed_data[35:38]

    d["title_a"] = get_record_values(record, "245$a")
    d["title_b"] = get_record_values(record, "245$b")
    d["title_c"] = get_record_values(record, "245$c")
    d["title_p"] = get_record_values(record, "245$p")

    d["title"] = " ".join([d["title_a"], d["title_b"], d["title_p"]])

    d["title_variation_a"] = get_record_values(record, "246$a")
    d["title_variation_b"] = get_record_values(record, "246$b")

    d["author_names"] = " ".join(
        [get_record_values(record, "100$a"), get_record_values(record, "700$a")]
    )
    d["corporate_names"] = " ".join(
        [get_record_values(record, "110$a"), get_record_values(record, "710$a")]
    )
    d["meeting_names"] = " ".join(
        [get_record_values(record, "111$a"), get_record_values(record, "711$a")]
    )

    d["publisher"] = record.publisher or ""

    d["pagination"] = get_record_values(record, "300$a")
    d["dimensions"] = get_record_values(record, "300$c")

    return d


def load_records(path):
    records = []
    extension = path.split(".")[-1]
    if extension == "mrc" or extension == "marc":
        with open(path, "rb") as marcfile:
            reader = pymarc.MARCReader(marcfile)
            records.extend(list(reader))
    elif extension == "json":
        with open(path, "r") as jsonfile:
            for line in jsonfile:
                record = pymarc.parse_json_to_array(line)[0]
                records.append(record)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    return records
