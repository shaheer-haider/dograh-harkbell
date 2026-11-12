"""What the agent had said when the caller cut it off stays in its context.

On 2026-11-10 a caller on a quiet handset said "Hello?" three times over the
opening of Harkbell's copy of Mirror Labs (run 4429), and each time the agent
started again from "Hi, this is Emma at Mirror Labs": the greeting played four
times. The prompt already told the model never to restart a sentence it was
cut off in, and to look back at its own lines first. It could not, because
there was nothing to look back at:

- A greeting (a configured text, a recording, or the model's own opening) is
  committed to the context only when it plays to the end
  (``GreetingController._finished``); one the caller talks over left nothing.
- A model reply cut off mid-sentence kept only the words already played, and
  only on a speech service that reports word timings. Without them a sentence
  reaches the context after its audio, so the sentence being spoken when the
  caller came in was lost with the rest.
- Neither said it had been cut off. "Hi, this is Emma at" reads to the model
  like a line it chose to end there.

So the model's next request held the prompt and "Hello?", and greeted.

Now what played stays, followed by ``CUT_OFF_MARKER``. When the exact extent
is not known, the sentences that had started playing are kept: with word
timings that is the words themselves; without them the sentence whose audio
had begun; for a recording, the sentences of its transcript that its played
audio had reached. The marker is kept out of what the caller hears and out of
the transcript: it is written into the context alone, and the speech filter
removes it if the model ever copies it into a reply (``tool_markup.py``).
"""

import re

from loguru import logger

from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    InterruptionFrame,
    LLMContextAssistantTimestampFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.string import TextPartForConcatenation
from pipecat.utils.run_context import run_id_var
from pipecat.utils.time import time_now_iso8601

# Short, and bracketed so a model reading its own history takes it as a fact
# about the line rather than as words it said. ``mark_cut_off`` puts a dash in
# front of it, which is how a transcript ends a sentence that was broken off.
CUT_OFF_MARKER = "[cut off by the caller]"

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")


def mark_cut_off(spoken: str) -> str:
    """The context's copy of a line the caller cut off: what played, then the marker."""
    return f"{spoken.rstrip()} — {CUT_OFF_MARKER}"


def commit_cut_off_line(context: LLMContext, spoken: str, *, since: int | None) -> None:
    """Add a cut-off line to the context where it was said.

    ``since`` is how many messages the context held when the line began to
    play. The caller's words that cut it off can reach the context first --
    their turn may end before the interruption has crossed the agent's own
    worker to the output -- and the line belongs in front of them, not after:
    a conversation ending in an assistant message reads to a model as its own
    unfinished reply to continue.
    """
    message = {"role": "assistant", "content": mark_cut_off(spoken)}
    # INFO, so production can see that it happened: run 4603's "Hello?" left
    # no trace in the engine's log and the question had to be answered from
    # the model's reply alone.
    logger.info(
        f"[run {run_id_var.get()}] Kept a line the caller cut off: "
        f"{len(spoken.strip())} chars"
    )
    messages = list(context.messages)
    position = len(messages)
    if since is not None and since <= position:
        while position > since and _is_user_message(messages[position - 1]):
            position -= 1
    if position == len(messages):
        context.add_message(message)
    else:
        context.set_messages([*messages[:position], message, *messages[position:]])


def _is_user_message(message) -> bool:
    return isinstance(message, dict) and message.get("role") == "user"


def sentences_started(text: str, played_fraction: float) -> str:
    """The sentences of ``text`` that ``played_fraction`` of it had reached.

    For speech whose timing is known only as a whole -- a recording and its
    transcript. Characters stand in for time: a sentence starting two thirds
    of the way through the transcript is taken to start two thirds of the way
    through the audio. Nothing played is nothing kept; any audio at all keeps
    the first sentence.
    """
    text = text.strip()
    if not text or played_fraction <= 0:
        return ""
    reached = min(played_fraction, 1.0) * len(text)
    starts = [0] + [match.end() for match in _SENTENCE_END.finditer(text)]
    started = [start for start in starts if start < reached]
    following = starts[len(started)] if len(started) < len(starts) else len(text)
    return text[:following].strip()


def _part(frame: AggregatedTextFrame) -> TextPartForConcatenation:
    return TextPartForConcatenation(
        frame.raw_text or frame.text,
        includes_inter_part_spaces=frame.includes_inter_frame_spaces,
    )


class StartedSentence:
    """The sentence whose audio may be playing, until its words arrive.

    A speech service announces each sentence (an ``AggregatedTextFrame`` it
    is about to speak) in front of its audio, and reports what was spoken
    afterwards: word by word as they play when it has word timings, or the
    whole sentence once its audio has drained when it has not. Between the
    two, only this knows the sentence had begun.
    """

    def __init__(self):
        self.part: TextPartForConcatenation | None = None
        self.context_id: str | None = None
        self.heard = False
        # Once the speech service has reported a word, it reports every word
        # as it plays and the spoken text is exact: its sentence
        # announcements are not consulted again. They can run ahead of the
        # audio there -- production run 4603's reply to "Hello?" was handed a
        # greeting sentence that had not begun to play.
        self.word_timed = False

    def observe(self, frame: Frame) -> None:
        if isinstance(frame, TTSTextFrame):
            # Its words (or the sentence itself) reached the listener: from
            # here the spoken text is exact.
            if frame.aggregated_by == "word":
                self.word_timed = True
            self.clear()
        elif (
            isinstance(frame, AggregatedTextFrame)
            and frame.will_be_spoken
            and not self.word_timed
        ):
            self.part = _part(frame)
            self.context_id = frame.context_id
            self.heard = False
        elif isinstance(frame, OutputAudioRawFrame) and frame.audio and self.part:
            self.heard = True

    def clear(self) -> None:
        self.part = None
        self.context_id = None
        self.heard = False


class HarkbellAssistantAggregator(LLMAssistantAggregator):
    """The assistant aggregator, keeping a cut-off reply as said and marked so."""

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(context, **kwargs)
        self._started = StartedSentence()
        # The speech now playing, and whether its text is meant for the
        # context at all (its TTSStartedFrame says): a transition notice or a
        # greeting is not, and a sentence of it must not be committed here.
        self._playing_context_id: str | None = None
        self._playing_appends = False
        self._cut_off = False
        # How long the context was when the reply began to play.
        self._turn_mark: int | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSStartedFrame):
                self._playing_context_id = frame.context_id
                self._playing_appends = frame.append_to_context
            else:
                self._started.observe(frame)
        await super().process_frame(frame, direction)

    async def reset(self):
        await super().reset()
        self._started.clear()

    async def _trigger_assistant_turn_started(self):
        self._turn_mark = len(self._context.messages)
        await super()._trigger_assistant_turn_started()

    async def _handle_interruptions(self, frame: InterruptionFrame):
        if self._assistant_turn_start_timestamp:
            started = self._started
            if (
                started.part is not None
                and started.heard
                and self._playing_appends
                and started.context_id == self._playing_context_id
            ):
                self._aggregation.append(started.part)
            self._cut_off = any(part.text.strip() for part in self._aggregation)
        try:
            await super()._handle_interruptions(frame)
        finally:
            self._cut_off = False
            self._started.clear()

    async def push_aggregation(self) -> str:
        """Upstream's push, with the marker on the context's copy of a cut-off line.

        The returned text -- what the transcript and the turn events carry --
        stays as it was said.
        """
        if not (self._cut_off and self._aggregation):
            return await super().push_aggregation()

        aggregation = self.aggregation_string()
        await self.reset()

        if aggregation and self._params.correct_aggregation_callback:
            try:
                aggregation = self._params.correct_aggregation_callback(aggregation)
            except Exception as e:
                logger.error(f"Error in aggregation correction callback: {e}")

        if aggregation.strip():
            logger.debug(f"{self} keeping a cut-off reply: {aggregation!r}")
            commit_cut_off_line(self._context, aggregation, since=self._turn_mark)

        await self.push_context_frame()
        await self.push_frame(LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601()))
        return aggregation


def harkbell_context_aggregators(
    context: LLMContext,
    *,
    user_params: LLMUserAggregatorParams,
    assistant_params: LLMAssistantAggregatorParams,
) -> tuple[LLMUserAggregator, HarkbellAssistantAggregator]:
    """A cascade call's aggregator pair, with ``HarkbellAssistantAggregator``.

    Built through Pipecat's own pair so the user half and the back-reference
    between the two are wired exactly as upstream wires them.
    """
    pair = LLMContextAggregatorPair(
        context,
        user_params=user_params,
        assistant_params=assistant_params,
        realtime_service_mode=False,
    )
    user = pair.user()
    assistant = HarkbellAssistantAggregator(
        context,
        params=assistant_params,
        _realtime_service_mode=False,
        _paired_user_aggregator=user,
    )
    return user, assistant
