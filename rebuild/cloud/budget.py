"""Upper-bound cost estimate from a saved Pricing API quote. A budget alert is not a cap."""
import argparse
import json


def estimate(hours, hourly, volume_gib, ebs_gib_month, s3_gib, s3_gib_month, extra):
    compute = hours*hourly
    ebs = volume_gib*ebs_gib_month*max(hours, 24)/730  # include stopped-disk days before teardown
    s3 = s3_gib*s3_gib_month
    return {'compute': compute, 'ebs': ebs, 's3': s3, 'extra': extra, 'total': compute+ebs+s3+extra}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('hours', 'hourly', 'volume-gib', 'ebs-gib-month', 's3-gib', 's3-gib-month', 'extra'):
        p.add_argument('--'+key, type=float, required=True)
    a = p.parse_args()
    print(json.dumps(estimate(a.hours, a.hourly, a.volume_gib, a.ebs_gib_month, a.s3_gib, a.s3_gib_month, a.extra), indent=2))
