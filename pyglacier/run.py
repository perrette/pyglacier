#!/home/perrette/anaconda/bin/python
"""Run the glacier model
"""
import argparse
import warnings
import subprocess
from collections import OrderedDict
import sys, os, json
from .namelist import Namelist, Params, Param
from .settings import CODEDIR, NML as NML_DEFAULT, OUTDIR as OUT_DIR, INFILE as IN_FILE, RSTFILE as RST_FILE

# print 'CODEDIR', CODEDIR
EXE = os.path.join(CODEDIR, 'main.exe') # default exe
# EXE="./main.exe"
# NML_EXPERIMENT = 'out/{name}/params.nml'

# autocomplete={'control':'ctr', 'geometry':'geo', 'dynamics':'dyn', 'surfacemb':'smb', 'fjordmelt':'fjo', 'basalmelt':'bas', 'calving':'cal'}

def get_checksum(codedir=CODEDIR, interactive=False):
    """ return git's checksum
    """
    cmd = 'cd {codedir} && git log | head -1 | sed "s/commit //"'.format(codedir=codedir)
    commit = subprocess.check_output(cmd, shell=True)

    cmd = 'cd {codedir} && git status | grep "Changes.*commit" \
        || dummy_command_to_avoid_error=1'.format(codedir=codedir)

    # cmd = 'cd {codedir} && echo $status | grep "Changes.*commit"'.format(codedir=codedir)
    changes = subprocess.check_output(cmd, shell=True)
    if changes != "":
        if interactive:
            cmd = 'cd {codedir} && git status'.format(codedir=codedir)
            os.system(cmd)
            y = raw_input("git directory not clean, proceed? (press 'y') ")
            if y != 'y':
                print "Stopped by user."
                sys.exit()
        else:
            print "git directory not clean"


    return commit

def write_namelist(nml_out, params=None, nml_in=NML_DEFAULT):
    nml_params = Namelist.read(nml_in)
    params = params or Params()
    # print params
    for p in params:
        if p in nml_params:
            nml_params[nml_params.index(p)] = p # just replace it
        else:
            nml_params.append(p)

    # write down the updated glacier namelist
    print "write namelist to",nml_out
    nml_params.write(nml_out)

def run_model(years, params=None, nml=NML_DEFAULT, exe=EXE, out_dir=OUT_DIR, in_file=IN_FILE, rst_file=None, continue_simu=False, cmd_args="", **kwargs):

    # create directory if it does not yet exists
    if not os.path.exists(out_dir):
        print "create directory "+out_dir
        os.makedirs(out_dir)

    # Indicate git check sum
    try:
        checksum = get_checksum(codedir=os.path.dirname(exe), interactive=False)
        with open(os.path.join(out_dir,'gitchecksum'),'w') as f:
            f.write(checksum)
    except Exception as error:
        warnings.warn("GIT CHECKING: "+error.message)

    # And the command
    with open(os.path.join(out_dir,'command'),'w') as f:
        f.write(" ".join(sys.argv))

    nml_experiment = os.path.join(out_dir, 'params.nml')
    write_namelist(nml_experiment, params=params, nml_in=nml)

    def _fmt_cmd_str(fn):
        # problem with ioparams
        return """ '"'{}'"' """.format(fn)

    kwargs['years'] = years
    kwargs['nml'] = nml
    kwargs['out_dir'] = _fmt_cmd_str(out_dir)
    kwargs['in_file'] = _fmt_cmd_str(in_file)
    kwargs['continue'] = 'T' if continue_simu else 'F'
    if rst_file:
        kwargs['rst_file'] = _fmt_cmd_str(rst_file)

    # passed to executable as command line
    cmd_args += " " + " ".join(["--{} {}".format(k, kwargs[k]) for k in kwargs])

    # execute the fortran script
    cmd = "{exe} {cmd}".format(exe=exe, cmd=cmd_args)
    print cmd
    return os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default=EXE, help="fortran executable")

    group = parser.add_argument_group("Alias for --cmd '...' ")
    group.add_argument("--nml", default=NML_DEFAULT, help="input namelist file")
    group.add_argument("-o","--out-dir", default=OUT_DIR, help="output directory")
    group.add_argument("-i","--in-file", default=IN_FILE, help="input file")
    group.add_argument("-r","--rst-file", default=RST_FILE, help="restart file")
    group.add_argument("-y","--years", default=100, type=float, help="years of simulation")
    group.add_argument("-c","--continue_simu", action="store_true", help="continue previous simulation")
    parser.add_argument("--cmd", default="", help="pass on arguments to fortran program (e.g. --cmd '--help'")

    group = parser.add_argument_group("Model parameter: update namelist")
    group.add_argument("--json", help="update glacier parameters (syntax: '{...}'")
    group.add_argument("--json-file", help="same as json but indicate a file")

    args = parser.parse_args()

    # update from command line parameters?
    if args.json or args.json_file:
        if args.json:
            params=json.loads(args.json)
        else:
            with open(args.json_file) as f:
                params=json.load(f)
    else:
        params = None

    response = run_model(args.years, continue_simu=args.continue_simu, params=params, nml=args.nml, exe=args.exe, out_dir=args.out_dir, in_file=args.in_file, rst_file=args.rst_file, cmd_args=args.cmd)
    print 'JOB IS DONE: ', response

if __name__ == "__main__":
    main()
