"""
Functions for performing SPICE-based electrical simulation of a neuron given
its synapses and skeleton using a simple linear passive model.


Try the `interactive simulation tutorial`_ for a tour of basic simulation options.

.. _interactive simulation tutorial: notebooks/SimulationTutorial.ipynb

.. note::

    The ``simulation`` module depends on additional packages.
    Install them from ``conda-forge``:

    .. code-block:: bash

       conda install -c conda-forge ngspice umap-learn scikit-learn matplotlib
"""
# Author: Stephen Plaza
# Delay modeling and spice parsing adapted from code  by Louis K. Scheffer.

import os
import math
import platform
from tempfile import mkstemp
from subprocess import Popen, PIPE, DEVNULL

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree

from .utils import tqdm, UMAP
from .client import inject_client
from .queries import fetch_synapse_connections

# Axon resistance.
Ra_LOW = 0.4
Ra_MED=1.2
Ra_HIGH=4.0

# Membrane resistance.
Rm_LOW = 0.2
Rm_MED=0.8
Rm_HIGH=3.11


class TimingResult:
    def __init__(self, bodyid, delay_matrix, amplitude_matrix, neuron_io, neuron_conn_info, symmetric=False):
        """
        Timing result constructor.

        Provides methods for parsing timing results.

        Args:

            bodyid (int):
                Segment id for neuron

            delay_matrix (dataframe):
                nxm matric of source to sink (if source set == sink set,
                the data can be used to cluster the provided io into different domains.

            neuron_io (dataframe):
                synapse information: location, ispre, brain region
        """
        self.bodyid = bodyid
        self.delay_matrix = delay_matrix
        self.amplitude_matrix = amplitude_matrix
        self.neuron_io = neuron_io
        self.symmetric = symmetric
        self.neuron_conn_info = neuron_conn_info

        # sources = set(delay_matrix.index.to_list())
        # sinks = set(delay_matrix.columns.values())
        # if sources == sinks:
        #     self.symmetric = True

    def compute_region_delay_matrix(self):
        """
        Generate delay and amplitude matrix based on primary brain regions.

        Averages the delay and amplitude between the sources and sinks of the
        timing results.

        Returns:
            (dataframe, dataframe) for the delay and amplitude from brain region to brain region.
        """
        assert(not self.symmetric)

        # determine row and column names
        inrois = set(self.neuron_io[self.neuron_io["io"] == "in"]["roi"].to_list())
        outrois = set(self.neuron_io[self.neuron_io["io"] == "out"]["roi"].to_list())

        inrois = list(inrois)
        outrois = list(outrois)

        inrois.sort()
        outrois.sort()

        delay_matrix = np.zeros((len(inrois), len(outrois)))
        amp_matrix = np.zeros((len(inrois), len(outrois)))

        roi_count = {}
        roi_delays = {}
        roi_amps = {}

        roi2index_in = {}
        roi2index_out = {}
        for idx, roi in enumerate(inrois):
            roi2index_in[roi] = idx
        for idx, roi in enumerate(outrois):
            roi2index_out[roi] = idx

        # roi info
        for drive, row in self.delay_matrix.iterrows():
            inroi = self.neuron_io[self.neuron_io["swcid"] == drive].iloc[0]["roi"]
            for out, val in row.items():
                outroi = self.neuron_io[self.neuron_io["swcid"] == out].iloc[0]["roi"]
                if (inroi, outroi) not in roi_delays:
                    roi_delays[(inroi, outroi)] = 0
                    roi_count[(inroi, outroi)] = 0
                roi_delays[(inroi, outroi)] += val
                roi_count[(inroi, outroi)] += 1

        # calculate average
        for drive, row in self.amplitude_matrix.iterrows():
            inroi = self.neuron_io[self.neuron_io["swcid"] == drive].iloc[0]["roi"]
            for out, val in row.items():
                outroi = self.neuron_io[self.neuron_io["swcid"] == out].iloc[0]["roi"]
                if (inroi, outroi) not in roi_amps:
                    roi_amps[(inroi, outroi)] = 0
                roi_amps[(inroi, outroi)] += val

        for key, val in roi_count.items():
            roi_in, roi_out = key
            idx1 = roi2index_in[roi_in]
            idx2 = roi2index_out[roi_out]
            delay_matrix[idx1, idx2] = roi_delays[key] / val
            amp_matrix[idx1, idx2] = roi_amps[key] / val

        # return dataframes
        return pd.DataFrame(delay_matrix, index=inrois, columns=outrois), pd.DataFrame(amp_matrix, index=inrois, columns=outrois)


    def plot_response_from_region(self, brain_region, path=None):
        """
        Show neuron response to brain regions based on inputs from a given region.

        Args:

            brain_region (str):
                Source brain region.
            path (str):
                Optionally save plot to designated file
        """
        assert(not self.symmetric)

        # matplot for amplitude and delay

        # filter sample brain region inputs
        inputs = self.neuron_io[(self.neuron_io["roi"] == brain_region) & (self.neuron_io["io"] == "in")]["swcid"].to_list()
        delay_matrix_sub = self.delay_matrix[self.delay_matrix.index.isin(inputs)]
        delay_matrix_sub = self.delay_matrix[self.delay_matrix.index.isin(inputs)]
        outrois = set(self.neuron_io[self.neuron_io["io"] == "out"]["roi"].to_list())

        delay_amp_region = []

        for drive, row in delay_matrix_sub.iterrows():
            row2 = self.amplitude_matrix.loc[drive]
            for out, val in row.items():
                outroi = self.neuron_io[self.neuron_io["swcid"] == out].iloc[0]["roi"]
                amp = row2[out]
                delay_amp_region.append([val, amp, outroi])

        plot_df = pd.DataFrame(delay_amp_region, columns=["delay", "amp", "region"])

        # create plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(8)
        for region in plot_df["region"].unique():
            tdata = plot_df[plot_df["region"] == region]
            ax.scatter(tdata["delay"].to_list(), tdata["amp"].to_list(), c=[np.random.rand(3,)], label=region)
        ax.legend()
        plt.xlabel("delay (ms)")
        plt.ylabel("amplitude (mV)")

        if path is not None:
            plt.savefig(path)

        plt.close()
        return fig


    def plot_neuron_domains(self, path=None):
        """
        Show how the different simulation points cluster in the neuron
        and their corresponding ROI.

        Plots the distance matrix in 2D using delay as the distance.

        Args:
            path (str):
                save plot to file as png
        """

        assert(self.symmetric)

        # ensure matrix is actually symmetric
        delays = self.delay_matrix.values
        for iter1 in range(len(self.delay_matrix)):
            for iter2 in range(iter1, len(self.delay_matrix)):
                if iter1 == iter2:
                    delays[iter1, iter2] = 0
                else:
                    val = (delays[iter1, iter2] + delays[iter2, iter1]) / 2
                    if val < 0:
                        val = 0
                    delays[iter1, iter2] = val
                    delays[iter2, iter1] = val

        # use umap to plot distance matrix
        u = UMAP(metric="precomputed", n_neighbors=90).fit(delays)
        points_2d = u.fit_transform(delays)

        # aasociate a region with each point
        x_y_region = []
        for idx, sid in enumerate(self.delay_matrix.index.to_list()):
            x_y_region.append([points_2d[idx][0], points_2d[idx][1], self.neuron_io[self.neuron_io["swcid"] == sid].iloc[0]["roi"]])

        plot_df = pd.DataFrame(x_y_region, columns=["x", "y", "region"])

        # create plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(8)
        for region in plot_df["region"].unique():
            tdata = plot_df[plot_df["region"] == region]
            ax.scatter(tdata["x"].to_list(), tdata["y"].to_list(), c=[np.random.rand(3,)], label=region)
        ax.legend()

        if path is not None:
            plt.savefig(path)

        plt.close()
        return fig


    def estimate_neuron_domains(self, num_components, plot=False):
        """
        Estimate the domains based on timing estimates.

        Note: only works for symmetric sink and source simulation.

        Args:
            num_components (int):
                number of cluster domains
            plot (bool):
                If True, create and return a plot that shows cluster results with predicted labels
                (can compare with roi labels from plot_neuron_domains).
                If ``plot`` is a string, it is interpreted as a filepath,
                to which the plot is also written to disk as a PNG.

        Returns:
            (dataframe, dataframe, plot) input and output connection summary split by domain,
            synapse-level neuron_io indicating component partition.
            If a ``plot`` was requested, then the generated plot is returned as the
            third tuple element, otherwise that element is ``None``.
        """

        assert(self.symmetric)

        # ensure matrix is actually symmetric
        delays = self.delay_matrix.values
        for iter1 in range(len(self.delay_matrix)):
            for iter2 in range(iter1, len(self.delay_matrix)):
                if iter1 == iter2:
                    delays[iter1, iter2] = 0
                else:
                    val = (delays[iter1, iter2] + delays[iter2, iter1]) / 2
                    if val < 0:
                        val = 0
                    delays[iter1, iter2] = val
                    delays[iter2, iter1] = val

        # create decomposition using hierarchical cluster over distance matrix
        Dsq = squareform(delays)
        cluster = linkage(Dsq, 'ward')

        if num_components > 1:
            partitions = cut_tree(cluster, n_clusters=num_components)
            best_labels = partitions[:,0]
        else:
            raise RuntimeError("must request 2 or more partitions")
            from sklearn.metrics import silhouette_score
            # find optimal number of clusters using silhouette function or based
            # on provided input
            best_score = -1
            best_labels = None

            for ncomps in range(2, len(self.delay_matrix)//4):
                partitions = cut_tree(cluster, n_clusters=ncomps)
                labels = partitions[:,0]
                score = silhouette_score(delays, labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_labels = labels

        fig = None
        if plot:
            # use umap to plot distance matrix
            u = UMAP(metric="precomputed", n_neighbors=90).fit(delays)
            points_2d = u.fit_transform(delays)

            # aasociate a region with each point
            x_y_region = []
            for idx, sid in enumerate(self.delay_matrix.index.to_list()):
                x_y_region.append([points_2d[idx][0], points_2d[idx][1], best_labels[idx] ])

            plot_df = pd.DataFrame(x_y_region, columns=["x", "y", "region"])

            # create plot
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            fig.set_figwidth(8)
            fig.set_figheight(8)
            for region in plot_df["region"].unique():
                tdata = plot_df[plot_df["region"] == region]
                ax.scatter(tdata["x"].to_list(), tdata["y"].to_list(), c=[np.random.rand(3,)], label=region)
            ax.legend()

            if isinstance(plot, str):
                plt.savefig(plot)

            plt.close()

        # build KD tree and associate synapses with each point after the cluster

        filter_list = []
        for drive, row in self.delay_matrix.iterrows():
            idx = self.neuron_io[self.neuron_io["swcid"] == drive].index[0]
            filter_list.append(idx)

        tree = cKDTree(self.neuron_io.iloc[filter_list]["coords"].to_list())
        neuron_conn_info = self.neuron_conn_info.copy()
        coords = list(zip(neuron_conn_info["x"], neuron_conn_info["y"], neuron_conn_info["z"]))
        neuron_conn_info["domain_id"] = [best_labels[tree.query(x)[1]] for x in coords]

        # build summary input / output table

        # find unique input and output groupings
        temp_df = neuron_conn_info.drop_duplicates(subset=["partner", "type", "domain_id"])
        summary_array = []
        for idx, row in temp_df.iterrows():
            io = "input"
            if row["type"] == "pre":
                io = "output"

            matchgroup = neuron_conn_info[(neuron_conn_info["partner"] == row["partner"]) &
                    (neuron_conn_info["type"] == row["type"]) &
                    (neuron_conn_info["domain_id"] == row["domain_id"])]
            count = len(matchgroup)
            rois = list(matchgroup["roi"].unique())
            summary_array.append([io, row["partner"], count, row["domain_id"], rois])
        connection_summary = pd.DataFrame(summary_array, columns=["io", "partner", "weight", "domain_id", "rois"])

        connection_summary = connection_summary.sort_values(by=["io", "weight"], ascending=[False, False]).reset_index(drop=True)
        return connection_summary, neuron_conn_info, fig

class NeuronModel:

    @inject_client
    def __init__(self, bodyid, Ra=Ra_MED, Rm=Rm_MED, Cm=1e-2, *, client=None):
        """
        Neuron model constructor.

        Create model of a neuron which can be simulated.
        in different ways.

        Args:

            bodyid (int):
                Segment id for neuron.
            Ra (float):
                axon resistance
                Examples: 0.4 (``Ra_LOW``), 1.2 (``Ra_MED``), 4.0 (``Ra_HIGH``)
            Rm (float):
                membrane resistance
                Examples: 0.2 (``Rm_LOW``), 0.8 (``Rm_MED``), 3.11 (``Rm_HIGH``)
            Cm (float):
                membrane capacitance (should not very too much between neurons)
        """
        self.bodyid = bodyid


        with tqdm(total=100, disable=not client.progress) as pbar:
            # retrieve healed skeleton
            pbar.set_description("fetching skeleton")
            self.skeleton_df = client.fetch_skeleton(bodyid, heal=True)
            #print("Fetched skeleton")
            pbar.update(20)

            # extract inputs and outputs
            pbar.set_description("fetching output connections")
            outputs = fetch_synapse_connections(bodyid, client=client)
            pbar.update(35)
            pbar.set_description("fetching input connections")
            inputs = fetch_synapse_connections(None, bodyid, client=client)
            pbar.update(35)
            pbar.set_description("creating spice model")
            #print("Fetched synapse connections")

            # combine into one dataframe

            inputs["type"] = ["post"]*len(inputs)
            inputs = inputs[["type", "x_post", "y_post", "z_post", "roi_post", "bodyId_pre" ]].rename(columns={"x_post": "x", "y_post": "y", "z_post": "z", "roi_post": "roi", "bodyId_pre": "partner"})
            inputs["roi"].replace(np.nan, "none", inplace=True)

            input_pins = inputs[["roi"]].copy()
            input_pins["coords"] = list(zip(inputs["x"], inputs["y"], inputs["z"]))
            input_pins["io"] = ["in"]*len(inputs)

            outputs["type"] = ["pre"]*len(outputs)
            outputs = outputs[["type", "x_pre", "y_pre", "z_pre", "roi_pre", "bodyId_post" ]].rename(columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "roi_pre": "roi", "bodyId_post": "partner"})
            outputs["roi"].replace(np.nan, "none", inplace=True)

            output_pins = outputs[["roi"]].copy()
            output_pins["coords"] = list(zip(outputs["x"], outputs["y"], outputs["z"]))
            output_pins["io"] = ["out"]*len(outputs)

            self.neuron_conn_info = pd.concat([inputs, outputs]).reset_index(drop=True)
            self.io_pins = pd.concat([input_pins, output_pins]).reset_index(drop=True)

            #if len(self.io_pins) == 0:
            #    raise RuntimeError("neuron must have at least 1 inputs or output")

            self.Ra = Ra
            self.Rm = Rm
            self.Cm = 1e-2 # farads per square meeter

            # associate node with each input and output (this will be subsampled later)
            # build kdtree
            tree = cKDTree(list(zip(self.skeleton_df["x"], self.skeleton_df["y"], self.skeleton_df["z"])))
            # apply
            self.io_pins["swcid"] = self.io_pins["coords"].apply(lambda x: tree.query(x)[1]+1)


            # get voxelSize (8e-9) assume nanometers
            self.resolution = client.fetch_custom("MATCH (m :Meta) RETURN m.voxelSize").iloc[0][0][0] * 1e-9

            def build_spice_model():
                #F is the indefinite integral of the area of a cone going from radius r1 to r2 over length L
                def F(x, r1, r2, L):
                    return 2.0*math.pi*(r1*x + (r2-r1)*x**2/(2*L))

                #G is the indefinite integral of the resistance of a cone going from radius r1 to r2 over length L
                def G(x, r1, r2, L):
                    if (r1 != r2):
                        return -(1.0/math.pi) * (L/(r2-r1) * 1.0/(r1 + (r2-r1)*x/L))
                    else:
                        return (1.0/math.pi) * (1.0/r1**2) * x

                # build model
                cs = [0.0] * len(self.skeleton_df)
                rg = [1e30]* len(self.skeleton_df)
                rs = [[0, 0, 1.0] for i in range(len(self.skeleton_df))]   # N-1 Rs, first has none.  node, node, value

                for idx, fromrow in  self.skeleton_df.iterrows():
                    if idx == 0:
                        continue

                    # only one root, should be first entry
                    assert(fromrow["link"] != -1)

                    # row number = link - 1
                    parent = int(fromrow["link"]-1)
                    torow = self.skeleton_df.iloc[parent]

                    # compute axonal resistance
                    L = math.sqrt((fromrow["x"] - torow["x"])**2 + (fromrow["y"] - torow["y"])**2 + (fromrow["z"] - torow["z"])**2) * self.resolution

                    if L == 0:
                        print("L=0 - should not happen")
                        L = 1.0e-9   # set to 1 nm

                    r1 = fromrow["radius"] * self.resolution
                    r2 = torow["radius"] * self.resolution

                    # axonal resistance
                    res = (G(L, r1, r2, L) - G(0, r1, r2, L)) * self.Ra

                    # compute membrane resistance both to and from
                    area_from = F(L/2, r1, r2, L) - F(  0, r1, r2, L)   # Half of segment
                    c_from = area_from * self.Cm
                    rg_from = self.Rm / area_from
                    area_to   = F(L,   r1, r2, L) - F(L/2, r1, r2, L)   # other half
                    c_to   = area_to * self.Cm
                    rg_to = self.Rm / area_to
                    cs[idx] += c_from
                    rg[idx] = rg[idx] * rg_from / (rg[idx] + rg_from)  # in parallel
                    cs[parent] += c_to
                    rg[parent] = rg[parent] * rg_to / (rg[parent] + rg_to)  # in parallel
                    rs[idx][0] = idx
                    rs[idx][1] = parent
                    rs[idx][2] = res
                for i in range(len(cs)):
                    cs[i] = cs[i] * 1000.0  # Convert millisec responses to seconds, to make moments numerically better

                # write-out model string
                modelstr = ""
                for i in range(len(cs)):
                    modelstr += f"C{i+1} {i+1} 0 {cs[i]}\n" # grounded C
                    modelstr += f"RG{i+1} {i+1} 0 {rg[i]}\n" # grounded R membrane resistance
                    assert(rg[i] > 0)
                    if i > 0:
                        modelstr += f"R{i+1} {rs[i][0]+1} {rs[i][1]+1} {rs[i][2]}\n" # axonal resistance
                        assert(rs[i][2] > 0)

                return modelstr

            self.spice_model = build_spice_model()
            pbar.update(10)
            pbar.set_description("built model")
            #print("Built model")

    def _runspice(self, drive, unique_outs):
        """
        Run spice injecting current for a given input and return response for all outputs.

        Note: drive and unique_outs should be swc node ids for an input and output.

        Args:

            drive (int):
                id for input
            unique_outs (list):
                ids for outputs

        Returns:

            Dataframe (output ids, delay, amplitude)
        """

        # apply current at the specified input location
        drive_str = f"RDRIVE {drive} {len(self.skeleton_df)+1} 10000000000\n" # 0.1 ns conductance
        drive_str += f"V1 {len(self.skeleton_df)+1} 0 EXP(0 60.0 0.1 0.1 1.1 1.0 40)\n"
        drive_str += ".tran 0.1 40\n" # work from 0-10 ms (try 40)
        drive_str += ".options filetype=binary\n"

        # call command line spice simulator and write to temporary file
        fd, path = mkstemp()

        if platform.system() == "Windows":
            ngspice = "ngspice_con.exe"
        else:
            ngspice = "ngspice"

        # run ngspice
        try:
            p = Popen([ngspice, "-b", "-r", path], stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
        except FileNotFoundError as ex:
            msg = ("The 'ngspice' circuit simulation tool is not installed (or not on your PATH).\n\n"
                   "Please install it:\n\n"
                   "  conda install -c conda-forge ngspice\n\n")
            raise RuntimeError(msg) from ex

        title = 'FIBSEM simulation\n'
        data = title + self.spice_model + drive_str
        p.communicate(data.encode())

        """Read ngspice binary raw files. Return tuple of the data, and the
        plot metadata. The dtype of the data contains field names. This is
        not very robust yet, and only supports ngspice.

        # Example header of raw file
        # Title: rc band pass example circuit
        # Date: Sun Feb 21 11:29:14  2016
        # Plotname: AC Analysis
        # Flags: complex
        # No. Variables: 3
        # No. Points: 41
        # Variables:
        #         0       frequency       frequency       grid=3
        #         1       v(out)  voltage
        #         2       v(in)   voltage
        # Binary:
        """
        sim_results = None
        BSIZE_SP = 512 # Max size of a line of data; we don't want to read the
                        # whole file to find a line, in case file does not have
                        # expected structure.
        MDATA_LIST = [b'title', b'date', b'plotname', b'flags', b'no. variables', b'no. points', b'dimensions', b'command', b'option']
        with os.fdopen(fd, 'rb') as fp:
            plot = {}
            count = 0
            arrs = []
            plots = []
            while (True):
                try:
                    mdata = fp.readline(BSIZE_SP).split(b':', maxsplit=1)
                except:
                    raise RuntimeError("cannot parse spice output")
                if len(mdata) == 2:
                    if mdata[0].lower() in MDATA_LIST:
                        plot[mdata[0].lower()] = mdata[1].strip()
                    if mdata[0].lower() == b'variables':
                        nvars = int(plot[b'no. variables'])
                        npoints = int(plot[b'no. points'])
                        plot['varnames'] = []
                        plot['varunits'] = []
                        for varn in range(nvars):
                            varspec = (fp.readline(BSIZE_SP).strip()
                                       .decode('ascii').split())
                            assert(varn == int(varspec[0]))
                            plot['varnames'].append(varspec[1])
                            plot['varunits'].append(varspec[2])
                    if mdata[0].lower() == b'binary':
                        rowdtype = np.dtype({'names': plot['varnames'],
                                             'formats': [np.complex_ if b'complex'
                                                         in plot[b'flags']
                                                         else np.float_]*nvars})
                        # We should have all the metadata by now
                        arrs.append(np.fromfile(fp, dtype=rowdtype, count=npoints))
                        plots.append(plot)
                        fp.readline() # Read to the end of line
                else:
                    break

            # only one analysis
            sim_results = arrs[0]
        # delete file
        os.unlink(path)

        times = [0.0] * len(sim_results)
        # for each output (based on skeleton compartment id),
        # determine the delay to the max voltage and the width
        out_results = []

        for i in range(len(sim_results)):
            times[i] = sim_results[i]["time"]
            # parse out delay and amplitdue responses
        volts = [0.0] * len(sim_results)

        for idx in range(len(unique_outs)):
            # compartment id
            j = unique_outs[idx]

            for i in range(len(sim_results)):
                volts[i] = sim_results[i][j]

            #Now find max voltage, time at max voltag,
            maxv = 0.0
            maxt = -1
            for i in range(1,len(volts)-1):
                if volts[i-1] < volts[i] and volts[i] > volts[i+1]:
                    #interpolate to find max
                    v1 = volts[i-1] - volts[i]  #relative value at t1 (negative)
                    v2 = volts[i+1] - volts[i]  #same for t2
                    x1 = times[i-1] - times[i]
                    x2 = times[i+1] - times[i]
                    a = (v1 - v2*x1/x2)/(x1**2 - x1*x2)
                    b = (v1 - a*x1**2)/x1
                    deltax = -b/(2*a)
                    maxv = volts[i]
                    maxt = times[i] + deltax
                    break     # should be only one peak.
            assert(maxt >= 0)

            maxt -= 1.1   # subtract the peak time of the input
            for i in range(len(volts)):
                if volts[i] > maxv/2:
                    break
            for k in reversed(range(len(volts))):
                if volts[k] > maxv/2:
                    break
            out_results.append([j, maxt, maxv, times[k] - times[i]])
            assert(maxt <= 20)

        return pd.DataFrame(out_results, columns=["comp id", "delay", "maxv", "width"])


    def simulate(self, max_per_region=10):
        """Simulate passive model based on neuron inputs and outputs.

        Args:

            max_per_region (int):
                maxinum number of inputs per primary ROI to sample (0 means all)

        Returns:

            TimingResult (contains input/output delay matrix)
        """


        # only grab unique skel comp id rows and randomize data
        unique_io = self.io_pins.drop_duplicates(subset=["swcid"]).sample(frac=1).reset_index(drop=True)

        # grab top max_per region per input and all outputs
        drive_list = []
        unique_outs = []

        # number of drives per input
        rcounter = {}
        for idx, row in unique_io.iterrows():
            if row["io"] == "out":
                unique_outs.append(row["swcid"])
            else:
                if row["roi"] not in rcounter:
                    rcounter[row["roi"]] = 0
                if max_per_region > 0 and rcounter[row["roi"]] == max_per_region:
                    continue
                rcounter[row["roi"]] += 1
                drive_list.append(row["swcid"])
        if len(unique_outs) == 0:
            raise RuntimeError("neuron must have at least one output")
        if len(drive_list) == 0:
            raise RuntimeError("neuron must specify at least one input")

        # simulate each drive (provide progress bar)
        delay_data = []
        amp_data = []
        for drive in tqdm(drive_list):
            # run simulation
            sim_results = self._runspice(drive, unique_outs)

            delay_data.append(sim_results["delay"].to_list())
            amp_data.append(sim_results["maxv"].to_list())

        delay_df = pd.DataFrame(delay_data, columns=unique_outs, index=drive_list)
        amp_df = pd.DataFrame(amp_data, columns=unique_outs, index=drive_list)

        # return simulation results
        return TimingResult(self.bodyid, delay_df, amp_df, self.io_pins, self.neuron_conn_info, False)


    def estimate_intra_neuron_delay(self, num_points=100):
        """Simulate delay between random parts of a neuron.

        This function produces a delay matrix for a set of points
        determined by pre or post-synaptic sites.  The result
        is a square distance matrix which can be clustered to
        determine isopotential regions.  The points where current
        are injected do not represent real neural mechanisms
        and in practice random points could be chosen for simulation.
        Synapse location are chosen for convenience and to ensure
        proper weighting on 'important' parts of the neuron.

        Args:

            num_points(int):
                number of points to simulate.

        Returns:

            TimingResult (contains input/output delay matrix)
        """
        if num_points < 10:
            raise RuntimeError("must specifiy at least 10 simulation points")

        # only grab unique skel comp id rows and randomize data
        unique_io = self.io_pins.drop_duplicates(subset=["swcid"]).sample(frac=1).reset_index(drop=True)

        # consider input/output response symmetrically to create a distance
        io_list = unique_io[unique_io["io"] == "out"][0:(num_points//2)]["swcid"].to_list()
        io_list.extend(unique_io[unique_io["io"] == "in"][0:(num_points//2)]["swcid"].to_list())

        # simulate each drive (provide progress bar)
        delay_data = []
        amp_data = []
        for drive in tqdm(io_list):
            # run simulation
            sim_results = self._runspice(drive, io_list)

            delay_data.append(sim_results["delay"].to_list())
            amp_data.append(sim_results["maxv"].to_list())

        delay_df = pd.DataFrame(delay_data, columns=io_list, index=io_list)
        amp_df = pd.DataFrame(amp_data, columns=io_list, index=io_list)

        # return simulation results
        return TimingResult(self.bodyid, delay_df, amp_df, self.io_pins, self.neuron_conn_info, True)
