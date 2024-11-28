import pytzer as pz


def test_name_switch():
    for name in ["CWTD23", "CHW22"]:
        pz.set_library(pz, name)
        assert pz.library.name == name


# test_name_switch()
