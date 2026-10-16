from .word_expert import WordExpert
from .format_expert import FormatExpert
from .meaning_expert import MeaningExpert
from .pattern_expert import PatternExpert
from .instruction_expert import InstructionExpert

CLASSES = {'word': WordExpert, 'format': FormatExpert, 'meaning': MeaningExpert,
           'pattern': PatternExpert, 'instruction': InstructionExpert}
