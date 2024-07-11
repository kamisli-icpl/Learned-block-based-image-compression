
import argparse
import json
import sys

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bjontegaard as bd

_backends = ["matplotlib", "plotly"]



def parse_json_file(filepath, metric):
  filepath = Path(filepath)
  name = filepath.name.split(".")[0]
  with filepath.open("r") as f:
    try:
      data = json.load(f)
    except json.decoder.JSONDecodeError as err:
      print(f'Error reading file "{filepath}"')
      raise err

    if "results" in data:
      results = data["results"]
    else:
      results = data

    if metric not in results:
      raise ValueError(
            f'Error: metric "{metric}" not available.'
            f' Available metrics: {", ".join(results.keys())}'
      )

    try:
      if metric == "ms-ssim":
        # Convert to db
        values = np.array(results[metric])
        results[metric] = -10 * np.log10(1 - values)

      return {
        "name": data.get("name", name),
        "xs": results["bpp"],
        "ys": results[metric],
      }
    except KeyError:
      raise ValueError(f'Invalid file "{filepath}"')



def setup_args():
  parser = argparse.ArgumentParser(description="")
  parser.add_argument(
    "-f",
    "--results-file",
    metavar="",
    default="",
    type=str,
    nargs=2,
    required=True,
  )
  parser.add_argument(
    "-m",
    "--metric",
    metavar="",
    type=str,
    default="psnr",
    help="Metric (default: %(default)s)",
  )    
  return parser


def main(argv):
  args = setup_args().parse_args(argv)

  scatters = []
  for f in args.results_file:
    rv = parse_json_file(f, args.metric)
    scatters.append(rv)


  #print(scatters)
  print(f"Number of results to compare: {len(scatters)}. Here are the results and their data points...")
  print(scatters[0])
  print(scatters[1])

  # Test data
  rate_anchor = scatters[0]['xs'][1:6]
  psnr_anchor = scatters[0]['ys'][1:6]
  rate_test = scatters[1]['xs']
  psnr_test = scatters[1]['ys']

  # Use the package
  bd_rate = bd.bd_rate(rate_anchor, psnr_anchor, rate_test, psnr_test, method='akima', require_matching_points=False, interpolators=False)
  bd_psnr = bd.bd_psnr(rate_anchor, psnr_anchor, rate_test, psnr_test, method='akima', require_matching_points=False, interpolators=False)
  #print(bd_rate)
  #print(bd_psnr)

  print(f"BD-Rate: {bd_rate:.4f} %")
  print(f"BD-PSNR: {bd_psnr:.4f} dB")


if __name__ == "__main__":
  main(sys.argv[1:])
