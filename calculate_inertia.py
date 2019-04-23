
import rmsd

import sys

import gzip
import numpy as np
import cheminfo

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

import matplotlib.pyplot as plt

import numpy.linalg as linalg

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

    rxx = Y**2 + Z**2
    ryy = X**2 + Z**2
    rzz = X**2 + Y**2

    Ixx = atoms*rxx
    Iyy = atoms*ryy
    Izz = atoms*rzz

    Ixy = atoms*Y*X
    Ixz = atoms*X*Z
    Iyz = atoms*Y*Z

    Ixx = np.sum(Ixx)
    Iyy = np.sum(Iyy)
    Izz = np.sum(Izz)

    Ixy = np.sum(Ixy)
    Ixz = np.sum(Ixz)
    Iyz = np.sum(Iyz)

    inertia = np.zeros((3,3))

    inertia[0,0] = Ixx
    inertia[1,1] = Iyy
    inertia[2,2] = Izz

    inertia[0,1] = -Ixy
    inertia[1,0] = -Ixy
    inertia[0,2] = -Ixz
    inertia[2,0] = -Ixz
    inertia[1,2] = -Iyz
    inertia[2,1] = -Iyz

    w, v = linalg.eig(inertia)

    return w


def get_inertia_diag(atoms, coordinates):

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


def parse_molobj_conf(molobj, nconf=1000, dumpcoord=False):

    atoms, coordinates = cheminfo.molobj_to_xyz(molobj)

    print("generating confs")

    conformers = cheminfo.molobj_conformers(molobj, nconf)

    print("generating confs, done")

    inertias = []

    atomsstr = [str(atom) for atom in atoms]
    dumpxyz = []

    for conformer in conformers:
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)
        inertia = get_inertia(atoms, coordinates)

        if dumpcoord:
            dumpxyz.append(rmsd.set_coordinates(atomsstr, coordinates))

        yield inertia

    if dumpcoord:
        dumpxyz = "\n".join(dumpxyz)
        f = open("dump.xyz", 'w')
        f.write(dumpxyz)
        f.close()


def clear_molobj(molobj, add_hydrogens=True, optimize=False):

    smiles = cheminfo.molobj_to_smiles(molobj)
    molobj, status = cheminfo.smiles_to_molobj(smiles)

    if molobj is None:
        return None

    conformers = cheminfo.molobj_conformers(molobj, 1)

    if add_hydrogens:
        molobj = Chem.AddHs(molobj)

    if optimize:
        status = cheminfo.molobj_optimize_mmff(molobj)
        if status > 0:
            return None

    try:
        molobj.GetConformer()

    except ValueError:
        return None

    return molobj


def parse_molobj(molobj, optimize=False, add_hydrogens=True):

    atoms, coordinates = cheminfo.molobj_to_xyz(molobj)

    # dxyz = rmsd.set_coordinates([str(atom) for atom in atoms], coordinates)
    # f = open("dump.xyz", 'a+')
    # f.write(dxyz)
    # f.write("\n")
    # f.close()

    inertia = get_inertia(atoms, coordinates)

    return inertia


def parse_xyz(filename):

    atoms, coordinates = rmsd.get_coordinates_xyz(filename)

    inertia = get_inertia(atoms, coordinates)

    return inertia


def parse_sdf(filename, nconf=1):

    suppl = Chem.SDMolSupplier(filename,
        removeHs=False,
        sanitize=True)

    for molobj in suppl:

        if molobj is None:
            continue

        if nconf is None:
            inertia = parse_molobj(molobj)
            yield inertia

        else:
            inertias = parse_molobj_conf(molobj, nconf=nconf)
            for inertia in inertias:
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


def parse_filename(filename, nconf=None, **kwargs):

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

        generator = parse_sdf(filename, nconf=nconf)

    elif fileext == "smi":

        generator = parse_smi(filename)

    else:

        print("error: don't know how to parse file")
        quit()

    return generator


def procs_parse_sdfgz(filename, procs=1, per_procs=None):

    with gzip.open(filename) as f:
        filetxt = f.read()

    filetxt = filetxt.decode()
    filetxt = filetxt.split("$$$$\n")

    results = procs_sdflist(filetxt, procs=procs, per_procs=per_procs)

    return results


def procs_parse_sdf(filename, procs=1, per_procs=None):

    with open(filename) as f:
        filetxt = f.read()

    filetxt = filetxt.split("$$$$\n")

    results = procs_sdflist(filetxt, procs=procs, per_procs=per_procs)

    return results


def procs_sdflist(sdf_list, procs=1, per_procs=None):

    import multiprocessing

    N = len(sdf_list)

    if per_procs is None:
        per_procs = np.ceil(float(N) / float(procs))
        per_procs = int(per_procs)

    jobs = [sdf_list[line:line+per_procs] for line in range(0, N, per_procs)]

    del sdf_list

    pool = multiprocessing.Pool(processes=procs)
    results_out = pool.map(worker_sdfstr, jobs)
    results_flat = [item for sublist in results_out for item in sublist]

    return results_flat


def worker_sdfstr(lines, append_smiles=False, add_hydrogen=True, optimize=True):

    result = []

    for line in lines:

        molobj = Chem.MolFromMolBlock(line, removeHs=False)

        if molobj is None: continue

        molobj = clear_molobj(molobj)

        if molobj is None: continue

        # if add_hydrogen:
        #     molobj = Chem.AddHs(molobj)
        #     if molobj is None: continue
        #
        # if optimize:
        #     try:
        #         status = cheminfo.molobj_optimize(molobj)
        #     except:
        #         continue

        inertia = parse_molobj(molobj)

        if append_smiles:
            smi = Chem.MolToSmiles(molobj)
            inertia = [smi] + list(inertia)

        result.append(inertia)

    return result


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help="Calculate inertia of filename.{.sdf.gz,.smi.gz,.sdf,smi}")
    parser.add_argument('-j', '--procs', type=int, help="Use subprocess to run over more cores", default=1)
    parser.add_argument('--ratio', action="store_true", help="calculate ratio")
    parser.add_argument('--nconf', type=int, help="how many conformers per compound", default=None)

    parser.add_argument('--prepend-smiles', action="store_true", help="")

    # TODO re-generate 3D coordinates from SDF (for chembl database)
    # sdf -> molobj -> smiles -> molobj -> add h -> inertia

    args = parser.parse_args()

    if args.procs > 1:
        generator = procs_parse_sdf(args.filename, procs=args.procs)

    elif args.filename:
        generator = parse_filename(args.filename, nconf=args.nconf)

    for result in generator:

        if args.ratio:
            result = get_ratio(result)
            fmt = "{:5.3f}"
        else:
            fmt = "{:15.8f}"

        result = [fmt.format(x) for x in result]

    return

if __name__ == "__main__":
    main()

