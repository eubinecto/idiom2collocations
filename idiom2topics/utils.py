from merge_idioms.builders import MIPBuilder
from spacy import Language


def load_mip() -> Language:
    mip_builder = MIPBuilder()
    mip_builder.construct()
    print("############## mip loaded ###############")
    return mip_builder.mip