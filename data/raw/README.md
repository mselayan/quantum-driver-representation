# Getting the data

The I-24 MOTION dataset is **not distributed with this repository**. It is
released by Vanderbilt under a data use agreement and must be downloaded from the
source.

## The exact file used in the paper

| | |
| --- | --- |
| Release | I-24 MOTION INCEPTION |
| Date | 22 November 2022, Tuesday |
| Recording start | 06:00 |
| Duration | 4 hours |
| Collection identifier | `637c399add50d54aa5af0cf4__post2` |

This is one day of the INCEPTION release, not the whole thing. Every result in
the paper comes from this single file.

## Download

1. Register at <https://i24motion.org> and wait to be verified.
2. Go to <https://i24motion.org/data>.
3. Download the 22 November 2022 file listed above.
4. Put it in this directory.

Then point the pipeline at it:

```bash
python scripts/00_preprocess/preprocess.py \
    --json data/raw/637c399add50d54aa5af0cf4__post2.json
```

The filename does not matter to the code. Nothing assumes one.

## Required citation

The data use agreement requires this citation in any published work using the
data:

> Gloudemans, D., Wang, Y., Ji, J., Zachar, G., Barbour, W., Hall, E.,
> Cebelak, M., Smith, L., and Work, D.B. (2023). I-24 MOTION: An instrument for
> freeway traffic science. *Transportation Research Part C: Emerging
> Technologies*, 155, 104311.

The agreement also prohibits any attempt to re-identify individuals in the
dataset.

## Required schema

The code reads the raw trajectory JSON: a list of objects, one per vehicle.

| field                  | type           | meaning                                   |
| ---------------------- | -------------- | ----------------------------------------- |
| `direction`            | int            | `-1` westbound, `1` eastbound             |
| `coarse_vehicle_class` | int            | `0` sedan, `1` midsize, `2` van, `3` pickup, and larger classes above |
| `first_timestamp`      | float          | UTC seconds                               |
| `last_timestamp`       | float          | UTC seconds                               |
| `timestamp`            | list of float  | per frame, 25 Hz                          |
| `x_position`           | list of float  | per frame, ft, longitudinal               |
| `y_position`           | list of float  | per frame, ft, lateral                    |
| `length`               | float          | ft. Optional; defaults to 5.0 if absent.  |

`timestamp`, `x_position` and `y_position` must all be the same length.

`preprocess.py` validates this on load and fails immediately with a readable
message if the file is the wrong format, the wrong direction, or missing a
field. It does not fail halfway through a long run.

Consult the official I-24 MOTION documentation for the authoritative field
definitions: <https://github.com/I24-MOTION/I24M_documentation>

## Both directions are in the one file

The downloaded file holds eastbound and westbound trajectories together. The code
keeps only westbound (`direction == -1`) and ignores the rest. There is no
separate westbound download and none is needed.

## Which vehicles are used

Two different filters, and the difference matters.

**Egos**, the drivers that get modeled, must have `coarse_vehicle_class` in
`[0, 1, 2, 3]` and be tracked for at least 10 s.

**Neighbors**, the surrounding traffic that determines headway, density and the
entropy terms, are **every westbound vehicle in the file**: every class, every
duration, no filter at all. A semi truck tracked for three seconds still blocks
the car behind it, so it still shapes that driver's context. It simply never gets
modeled itself.

So use the file as downloaded. Pre-filtering it will silently thin the
neighborhood and change every context variable.

## Size

The raw JSON is large. `preprocess.py` holds every westbound vehicle in memory to
build the spacetime index, and it rebuilds that index in each chunk. On a laptop,
use `--limit-egos` to score a subset; the index is still built from the full set
of vehicles either way. On a cluster, run the chunks as a job array.
