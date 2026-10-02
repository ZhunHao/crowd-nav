"""Write cloud/local/launch.json from discovered identifiers. Launches nothing."""
from pathlib import Path
import argparse
import json


def request(ami, instance_type, subnet, group, profile, root, run_id, hours, volume_gib):
    if not hours > 0 or not volume_gib > 0:
        raise ValueError('Positive wall time and volume size required')
    tags = [{'Key': 'Project', 'Value': 'shipnav'}, {'Key': 'RunId', 'Value': run_id},
            {'Key': 'MaxHours', 'Value': str(hours)}]
    return {'ImageId': ami, 'InstanceType': instance_type, 'MinCount': 1, 'MaxCount': 1,
            'SubnetId': subnet, 'SecurityGroupIds': [group], 'IamInstanceProfile': {'Name': profile},
            'MetadataOptions': {'HttpTokens': 'required', 'HttpEndpoint': 'enabled', 'HttpPutResponseHopLimit': 1},
            'BlockDeviceMappings': [{'DeviceName': root, 'Ebs': {'VolumeSize': int(volume_gib), 'VolumeType': 'gp3',
                                                                 'Encrypted': True, 'DeleteOnTermination': True}}],
            'InstanceInitiatedShutdownBehavior': 'stop', 'ClientToken': run_id,
            'TagSpecifications': [{'ResourceType': r, 'Tags': tags} for r in ('instance', 'volume')]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('ami', 'instance-type', 'subnet', 'group', 'profile', 'root', 'run-id'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--volume-gib', type=int, required=True)
    a = p.parse_args()
    Path('cloud/local').mkdir(parents=True, exist_ok=True)
    Path('cloud/local/launch.json').write_text(json.dumps(request(
        a.ami, a.instance_type, a.subnet, a.group, a.profile, a.root, a.run_id, a.hours, a.volume_gib), indent=2))
