
import sys

import gzip
import rmsd
import numpy as np
import cheminfo

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

import matplotlib.pyplot as plt


def center_of_mass(atoms, coordinates):

    total_mass = np.sum(atoms)

    X = coordinates[:,0]
    Y = coordinates[:,1]
    Z = coordinates[:,2]

    R = np.zeros(3)

    R[0] = np.sum(atoms*X)
    R[1] = np.sum(atoms*Y)
    R[2] = np.sum(atoms*Z)
    R /= total_mass

    return R


def get_inertia(atoms, coordinates):

    com = center_of_mass(atoms, coordinates)

    coordinates -= com

    X = coordinates[:,0]
    Y = coordinates[:,1]
    Z = coordinates[:,2]

    rx2 = Y**2 + Z**2
    ry2 = X**2 + Z**2
    rz2 = X**2 + Y**2

    Ix = atoms*rx2
    Iy = atoms*ry2
    Iz = atoms*rz2

    Ix = np.sum(Ix)
    Iy = np.sum(Iy)
    Iz = np.sum(Iz)

    inertia = np.zeros(3)
    inertia[0] = Ix
    inertia[1] = Iy
    inertia[2] = Iz

    return inertia


def get_ratio(inertia):

    inertia.sort()

    ratio = np.zeros(2)
    ratio[0] = inertia[0]/inertia[2]
    ratio[1] = inertia[1]/inertia[2]

    return ratio


def generate_structure(smiles):

    molobj = Chem.MolFromSmiles(smiles)
    molobj = Chem.AddHs(molobj)
    cheminfo.molobj_optimize(molobj)

    return molobj


def parse_molobj(molobj):

    atoms, coordinates = cheminfo.molobj_to_xyz(molobj)

    inertia = get_inertia(atoms, coordinates)

    return inertia


def parse_sdf(filename):

    suppl = Chem.SDMolSupplier(filename,
        removeHs=False,
        sanitize=True)

    for molobj in suppl:

        if molobj is None: continue

        inertia = parse_molobj(molobj)

        yield inertia


def parse_sdfgz(filename):

    f = gzip.open(filename)
    suppl = Chem.ForwardSDMolSupplier(f,
        removeHs=False,
        sanitize=True)

    for molobj in suppl:

        if molobj is None: continue

        inertia = parse_molobj(molobj)

        yield inertia


def parse_smi(filename, sep=None, idx=0):

    with open(filename) as f:

        for i, line in enumerate(f):

            if sep is None:
                line = line.strip().split()
            else:
                line = line.strip().split(sep)

            smi = line[idx]

            molobj = generate_structure(smi)

            if molobj is None: continue

            inertia = parse_molobj(molobj)

            yield inertia



def parse_smigz(filename, sep=None, idx=0):

    with gzip.open(filename) as f:

        for line in f:

            line = line.decode()

            if sep is None:
                line = line.strip().split()
            else:
                line = line.strip().split(sep)

            smi = line[idx]

            molobj = generate_structure(smi)

            if molobj is None: continue

            inertia = parse_molobj(molobj)

            yield inertia


def parse_filename(filename):

    fileext = filename.split(".")

    if fileext[-1] == "gz":
        fileext = ".".join(fileext[-2:])
    else:
        fileext = fileext[-1]

    if fileext == "sdf.gz":

        generator = parse_sdfgz(filename)

    elif fileext == "smi.gz":

        generator = parse_smigz(filename)

    elif fileext == "sdf":

        generator = parse_sdf(filename)

    elif fileext == "smi":

        generator = parse_smi(filename)

    else:

        print("error: don't know how to parse file")
        quit()

    return generator


def plot_scatter(dots):

    X = [0, 0.5, 1, 0]
    Y = [1, 0.5, 1, 1]
    plt.plot(X, Y)
    plt.plot(dots, "k.")

    return


def procs_parse_sdfgz(filename, procs=1, per_procs=None):

    import multiprocessing

    filetxt = gzip.open(filename).read()
    filetxt = filetxt.decode()
    filetxt = filetxt.split("$$$$\n")

    N = len(filetxt)

    if per_procs is None:
        per_procs = np.ceil(float(N) / float(procs))
        per_procs = int(per_procs)

    pool = multiprocessing.Pool(processes=procs)

    jobs = [filetxt[line:line+per_procs] for line in range(0, N, per_procs)]

    del filetxt

    results_out = pool.map(worker_sdfstr, jobs)
    results_flat = [item for sublist in results_out for item in sublist]

    return results_flat


def worker_sdfstr(lines):

    result = []

    for line in lines:

        molobj = Chem.MolFromMolBlock(line)

        if molobj is None: continue

        inertia = parse_molobj(molobj)
        ratio = get_ratio(inertia)

        result.append(ratio)

    return result


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help="Calculate inertia of filename.{.sdf.gz,.sdf,smi}")
    parser.add_argument('-j', '--procs', type=int, help="Use subprocess to run over more cores", default=1)
    args = parser.parse_args()

    if args.procs > 1:
        generator = procs_parse_sdfgz(args.filename, procs=args.procs)
        quit()

    elif args.filename:
        generator = parse_filename(args.filename)


    for inertia in generator:
        print(inertia)

    return

if __name__ == "__main__":
    main()

