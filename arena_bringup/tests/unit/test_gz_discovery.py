"""Per-domain gz-transport discovery ports."""

from __future__ import annotations

from arena_bringup.defaults import gz_discovery_env

DOMAINS = range(233)
DDS_FIRST_PORT = 7400


def test_every_domain_gets_its_own_port_pair():
    ports = [int(p) for d in DOMAINS for p in gz_discovery_env(d).values()]
    assert len(ports) == len(set(ports)) == 2 * len(DOMAINS)


def test_service_port_follows_the_message_port():
    for d in DOMAINS:
        env = gz_discovery_env(d)
        assert int(env["GZ_DISCOVERY_SRV_PORT"]) == int(env["GZ_DISCOVERY_MSG_PORT"]) + 1


def test_ports_stay_below_the_dds_blocks_and_the_gz_defaults():
    ports = [int(p) for d in DOMAINS for p in gz_discovery_env(d).values()]
    assert 1024 <= min(ports)
    assert max(ports) < DDS_FIRST_PORT
    assert not {10317, 10318} & set(ports)
