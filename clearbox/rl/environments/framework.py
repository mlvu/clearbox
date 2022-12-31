from abc import ABC, abstractmethod

class Environment(ABC):
    """
    Abstract base class to represent an environment. The object represents the current state of the environment, which
    is updated after an action is taken with `act(action)`.
    """

    # @classmethod
    # @abstractmethod
    # def make(cls, *args, **kwargs):
    #     """
    #     Create an environment instance for the given class.
    #     :return:
    #     """

    @abstractmethod
    def act(self, action):
        """
        Submit an action to the Environment (called by/on behalf of the agent).

        :return: reward, a numeric value representing the reward resulting for the action.
        """

    @abstractmethod
    def state(self):
        """
        A representation of the current state.

        If this environment represents a partially-observable process, then only the observable parts of the state are
        returned.
        """

    @abstractmethod
    def full_state(self):
        """
        Representation of the complete state. Not to be called by an agent (but may be useful for debugging)
        :return:
        """

    @abstractmethod
    def finished(self):
        """
        Whether the environment has been finished, for instance if a "game over" has been reached. For infinite-length
        environments, this always returns True.

        :return:
        """

class Agent(ABC):
    """
    Representation of an agent. This implementation should not attempt to record a stateful representation of the "current
    environment".
    """

    @abstractmethod
    def move(self, state):
        """
        Given the current state, return a move.

        :param state:
        :return:
        """
        pass

    # @abstractmethod
    # def observe(self, state, reward, oldstate, action):
    #     """
    #     Observe the result of taking `action` in `oldstate` in terms of the reward claimed and the new state.
    #
    #     TODO: this may not be flexible enough for all training setups. Perhaps just remove.
    #
    #     :param state:
    #     :param reward:
    #     :param action:
    #     :return:
    #     """




