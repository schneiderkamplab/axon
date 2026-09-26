from synapse.axon.codegen2_common import parameter_pack_bindings


def test_parameter_packs_resolve_repeated_multisegment_scopes_and_existing_outputs():
    spec = {
        "output": "packed.{scope}.q.{scope}.k",
        "inputs": ("{scope}.query.weight", "{scope}.key.weight"),
    }
    scope = "encoder.layer.0"
    output = f"packed.{scope}.q.{scope}.k"
    inputs = [f"{scope}.query.weight", f"{scope}.key.weight"]
    expected = [(output, inputs)]
    assert list(parameter_pack_bindings([*inputs, output], spec)) == expected
    assert list(parameter_pack_bindings([], spec, target_key=output)) == expected
    assert list(parameter_pack_bindings([], spec, target_key="packed.a.q.b.k")) == []
