"""Base class for a memory representation of any dataset"""


class Dataset:
    """Represents a dataset read to memory"""

    def __init__(self, feature_names=[]):
        """Initializes a new instance of Dataset

        Parameters
        ----------
        feature_names : list<str>, optional
            List of names of the features present in this dataset.
        """
        if type(feature_names) != list:
            raise TypeError(
                "The `feature_names` argument must be of type list")
        self.features = [str(name) for name in feature_names]
        self.data = []

    def add_datapoint(self, datapoint):
        """Adds a datapoint to the dataset

        Parameters
        ----------
        datapoint : list
            A list containing the feature values.
        """
        point = {}  # datapoint to be built and inserted in the dataset
        if len(self.features) == 0:  # in case there are no feature names
            if len(self.data) > 0 and len(self.data[0]) != len(datapoint):
                raise TypeError("The new datapoint must be of the same size " +
                                "as the other datapoints. The new datapoint " +
                                "has size " + str(len(datapoint)) + ", but " +
                                "the other datapoints have size " +
                                str(len(self.data[0])) + ".")
            i = 0
            for value in datapoint:
                point["feature" + str(i)] = self.__tryparse__(value)
                i += 1
            self.data.append(point)
            return
        if len(datapoint) != len(self.features):
            raise TypeError("The datapoint must be of the same size as " +
                            "the features list. The features list has size " +
                            str(len(self.features)) + " and the datapoint " +
                            "has size " + str(len(datapoint)) + ".")
        i = 0
        for feature_name in self.features:
            point[feature_name] = self.__tryparse__(datapoint[i])
            i += 1
        self.data.append(point)  # actually adds the datapoint to the set

    def __len__(self):
        """Returns the length of this dataset"""
        return len(self.data)

    def __iter__(self):
        """Iterates through the objects in this dataset"""
        return iter(self.data)

    def __getitem__(self, key):
        """Gets the dataset at the specified index"""
        if type(key) != int:
            raise TypeError("The index must be an integer, instead got " + key)
        return self.data[key]

    def __tryparse__(self, value):
        """Parses the value into int, float or string

        Parameters
        ----------
        value : str
            A value to be parsed.

        Returns
        -------
        val : int, float or str
            The value after being parsed to its correct type.

        Notes
        -----
        The value will be parsed in a try and error way. First, we try to cast
        it to int. If that fails, we try to cast it to float. And if that fails
        as well, we simply return it as string.
        """
        value = value.strip()
        if type(value) != str:
            return value
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
