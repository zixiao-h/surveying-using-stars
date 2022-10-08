import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Data:
    def __init__(self, data_table, name=None):
        """Initialise all variables unique to this particular dataset"""
        self.table = data_table
        self.telescopes = Telescopes(data_table)
        self.N_datapoints = data_table.shape[0]
        # sum of telescope coordinates/total number of pops/number of constant terms
        self.N_parameters = self.telescopes.N_telescope_positions + self.telescopes.N_pops # + self.telescopes.N_telescopes
        self.design = Design(self)
        self.svd = SVD(self)
        self.name = name


class Telescopes:
    """List of unique telescopes and POP settings for each defines a row in the design matrix uniquely"""
    def __init__(self, data_table):
        self.all_telescopes = ['W1', 'W2', 'S1', 'S2', 'E1', 'E2']
        self.unique_telescopes = self.all_telescopes.copy()
        self.pops_dictionary = {}
        self.pops_reference = []
        self.N_pops = 0
        self.N_telescopes = 0
        self.N_telescope_positions = 0
        self.G_pop = None

        # Make list of unique telescope names
        tel_1 = data_table['tel_1'].unique()
        tel_2 = data_table['tel_2'].unique()
        unique_telescopes_set = set().union(tel_1, tel_2)
        for telescope in self.all_telescopes:
            if telescope not in unique_telescopes_set:
                self.unique_telescopes.remove(telescope)

        # Construct dictionary showing possible POP settings for each telescope
        for telescope in self.unique_telescopes:
            # For a particular telescope:
            pops1 = set(data_table.loc[data_table['tel_1'] == telescope]['pop_1'].unique()) # unique pop1 settings
            pops2 = set(data_table.loc[data_table['tel_2'] == telescope]['pop_2'].unique()) # unique pop2 settings
            pops = set().union(pops1, pops2) # combine unique pop settings
            self.pops_dictionary[telescope] = list(pops) # add to dictionary as a list

        # Create count of total number of distinct POP settings
        for pop_list in self.pops_dictionary.values():
            self.N_pops += len(pop_list)
        
        # Create reference for order of POP settings in a row of the design matrix
        for telescope in self.unique_telescopes:
            for telescope_pop in self.pops_dictionary[telescope]:
                telescope_pop_id = telescope+'-'+telescope_pop # Create unique ID for a POP setting of a given telescope
                self.pops_reference.append(telescope_pop_id)
        
        self.N_telescopes = len(self.unique_telescopes)
        self.N_telescope_positions = 3*(self.N_telescopes-1)
    
        # Create graph of POP settings pairings
        df = data_table
        df['telpop_1'] = df.tel_1 + '-' + df.pop_1
        df['telpop_2'] = df.tel_2 + '-' + df.pop_2
        df_graph = df.groupby(['telpop_1', 'telpop_2']).size().reset_index().rename(columns={0:'count'})
        unique_telpops = set(df['telpop_1'].unique().tolist() + df['telpop_2'].unique().tolist())
        G = nx.Graph()
        G.add_nodes_from(unique_telpops)
        for index, row in df_graph.iterrows():
            G.add_edge(row['telpop_1'], row['telpop_2'])
        self.G_pop = G
    
    
    def indices_of(self, given_telescope, given_pop):
        """For a given telescope and POP setting, return the index of the telescope x coordinate,
        the index of the POP setting, and the index of the constant term within a row of the design matrix."""
        
        # Find index of x-coordinate of the telescope
        telescope_index = self.unique_telescopes.index(given_telescope)
        if given_telescope == 'W1':
            # Return None if telescope is W1, since W1 is used as the reference for the baseline vectors
            x_index = None
        else:
            x_index = 3*(telescope_index-1)
        
        # Use the reference list to calculate the index of the given POP setting
        given_pop_id = given_telescope+'-'+given_pop
        pop_index = self.N_telescope_positions + self.pops_reference.index(given_pop_id)

        # const_index = self.N_telescope_positions + self.N_pops + telescope_index
        
        return x_index, pop_index #, const_index
    

    def get_index(self, telescope):
        """Returns position index of telescope within unique_telescopes - 1, i.e. W2 is index 0, for referencing purposes"""
        if telescope == "W1":
            raise Exception("W1 is the reference telescope")

        return self.unique_telescopes.index(telescope) - 1
    

    def draw_pops(self, labels=True, size=(10,10), fontsize=15):
        """Plots graph of POP pairings"""
        plt.figure(figsize=size)
        nx.draw(self.G_pop, with_labels=labels, node_size=40)
        plt.title(f'POP settings pairings for {self.data.name}', font='serif', fontsize=fontsize)
    

    def pops_linked(self, telpop, querypop=None):
        """Returns set of pops paired with a given telpop, returns True/False if querypop given is linked to telpop"""
        components = nx.node_connected_component(self.G_pop, telpop)
        if querypop != None:
            return (querypop in components)
        else:
            return components


class Design:
    """Data structure that calculates and stores the design matrix and data vector 
    from a data object"""
    def __init__(self, data):
        """<data> must be a Data object"""
        self.data = data
        self.A = np.zeros((data.N_datapoints, data.N_parameters)) # initialise design matrix
        self.b = np.array(data.table['cart_1'] - data.table['cart_2']) # create data vector 

        # Construct design matrix
        for i in range(data.N_datapoints):
            data_row = data.table.iloc[i]
            design_row = np.zeros(data.N_parameters)

            # Calculate unit vector S to star
            phi = np.deg2rad(data_row['azimuth'])
            theta = np.deg2rad(data_row['elevation'])
            S = [np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi), np.sin(theta)] # unit vector pointing towards star
            
            # Get location of nonzero coefficients
            telescope_1 = data_row['tel_1']
            telescope_2 = data_row['tel_2']
            pop_1 = data_row['pop_1']
            pop_2 = data_row['pop_2']
            x_index_1, pop_index_1 = data.telescopes.indices_of(telescope_1, pop_1)
            x_index_2, pop_index_2 = data.telescopes.indices_of(telescope_2, pop_2)

            for j in range(3):
                # If one of the telescopes is W1, leave S = 0 since coordinates are zero
                if telescope_1 != 'W1':
                    design_row[x_index_1 + j] = S[j]
                if telescope_2 != 'W1':
                    design_row[x_index_2 + j] = -S[j]

            design_row[pop_index_1] = -1
            design_row[pop_index_2] = 1

            #design_row[const_index_1] = -1
            #design_row[const_index_2] = 1

            self.A[i] = design_row


class SVD:
    """Class containing methods to calculate model parameters, uncertainties etc.
    using SVD"""
    def __init__(self, data):
        self.data = data
        self.A = data.design.A
        self.b = data.design.b
        self.u, self.w, self.vh = np.linalg.svd(self.A, full_matrices=False) # since design matrix is not square (overdetermined)

        # Calculate w_inv; if any w/w_max < this min ratio, corresponding w_inv is taken to be zero
        machine_precision = np.finfo(np.float64).eps
        min_sv_ratio = self.data.N_datapoints * machine_precision
        w_inv = 1/self.w
        w_inv[self.w < self.w.max() * min_sv_ratio] = 0 # replace singular values under min ratio in 1/w with zero
        self.w_inv = w_inv


    def x(self, variables='telescopes', telescope=None):
        """Return model parameters vector x as calculated by SVD, removing degeneracies (singular values near zero).
        By default returns positions only; variables='pops' if want pops, set telescope='W2' etc if want position of specific telescope"""

        x = self.vh.T @ np.diag(self.w_inv) @ self.u.T @ self.b # vector of all parameters

        if variables == 'telescopes':
            # Just return positions
            N_telescope_positions = self.data.telescopes.N_telescope_positions
            N_telescopes = self.data.telescopes.N_telescopes
            positions = x[:N_telescope_positions].reshape(N_telescopes-1, 3) # since W1 is not included
            
            if telescope == None:
                return positions
            else:
                telescope_index = self.data.telescopes.get_index(telescope) # W1 not included as index
                return positions[telescope_index]
        
        elif variables == 'pops':
            return x[self.data.telescopes.N_telescope_positions:]
        
        elif variables == 'all':
            return x

        else:
            print('Incorrectly specified parameters')

    
    def residuals(self):
        """Returns list of residuals (differences between actual data vector b and the predicted values
        from the model, i.e. A multiplied by x)"""
        
        x = self.x(variables='all')
        residuals = self.data.design.A @ x - self.data.design.b # theoretical - observed

        return residuals

    
    def variances(self):
        """Return vector of variances for the estimated model parameters. Set std=True if want standard deviations, 
        all_parameters=True if want all variances instead of just variances of positions"""
        residuals = self.residuals()

        degrees_of_freedom = self.data.N_datapoints - self.data.N_parameters

        residualsquaresum = (residuals**2).sum() 
        rms_variance = residualsquaresum/degrees_of_freedom # RMS error on each measurement

        w_inv = np.reshape(self.w_inv, (-1, self.data.N_parameters)) # reshape w_inv into 2D array
        elements = (self.vh*w_inv.T)**2 * rms_variance # matrix to be summed over to find model parameters
        model_parameters_variance = elements.sum(axis=0)

        return model_parameters_variance

        
    def std(self, variables='telescopes', telescope=None, telpop=None):
        """Returns standard deviation of model parameters with option to return only those for
        positions, pops, or a specific delay from those"""

        all_std = np.sqrt(self.variances())
        telescopes = self.data.telescopes

        if variables == 'telescopes':
            telescope_std = all_std[:telescopes.N_telescope_positions].reshape(telescopes.N_telescopes-1, 3)
            if telescope == None:
                return telescope_std
            else:
                telescope_index = self.data.telescopes.get_index(telescope) # W1 not included
                return telescope_std[telescope_index]

        elif variables == 'pops':
            pops_std = all_std[telescopes.N_telescope_positions:]
            if telpop == None:
                return pops_std
            else:
                return pops_std[telescopes.pops_reference.index(telpop)]

        else:
            return all_std


    def positions_dict(self, telescope_name=None):
        """Returns a dictionary of the values of the telescope positions, with option to return specific telescope"""
        positions_dict = {}

        for telescope in self.data.telescopes.unique_telescopes[1:]:
            positions_dict[telescope] = self.x(telescope=telescope)

        if telescope_name != None:
            return positions_dict[telescope_name]
        else:
            return positions_dict
    

    def positions_array(self):
        """Returns array of telescope positions, in order specified by all_telescopes, zeroes included if telescope
        not in unique_telescopes"""
        positions_array = np.zeros((5,3))
        all_telescopes = self.data.telescopes.all_telescopes
        unique_telescopes = self.data.telescopes.unique_telescopes
        for i, telescope in enumerate(all_telescopes[1:]):
            if telescope in unique_telescopes:
                positions_array[i] = self.positions_dict(telescope)
        return positions_array


    def pops_values(self, pop_name=None):
        """Returns a dictionary of the values of the POP delays, with the option to return a specific delay"""
        x_pops = self.x(variables='pops')
        pops_values = {}

        for j, pop in enumerate(self.data.telescopes.pops_reference):
            pop_delay = x_pops[j]
            pops_values[pop] = pop_delay

        if pop_name != None:
            return pops_values[pop_name]

        return pops_values
