# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial

import pytest

from sparseml.core.event import Event, EventType
from sparseml.core.lifecycle.event import (
    CallbacksEventLifecycle,
    EventLifecycle,
    WrappedOptimEventLifecycle,
)


def test_event_lifecycle_abstract_class_can_not_be_instantiated():
    # tests event lifecycle abstract class can not be instantiated
    # directly, without implementing the abstract methods

    with pytest.raises(TypeError):
        EventLifecycle(type_first=EventType.BATCH_START, start=Event())


class EventLifecycleDummyChild(EventLifecycle):
    def batch_start_events(self):
        return [], "batch_start_events"

    def loss_calculated_events(self):
        return [], "loss_calculated_events"

    def optim_pre_step_events(self):
        return [], "optim_pre_step_events"

    def optim_post_step_events(self):
        return [], "optim_post_step_events"

    def batch_end_events(self):
        return [], "batch_end_events"


def _get_event_lifecycle(
    start=None, type_first=None, lifecycle_class=EventLifecycleDummyChild
):
    start = start or Event()
    type_first = type_first or EventType.BATCH_START
    lifecycle = lifecycle_class(type_first=type_first, start=start)
    return lifecycle


class TestEventLifecycle:
    @pytest.mark.parametrize("type_first", [EventType.BATCH_START, EventType.BATCH_END])
    @pytest.mark.parametrize(
        "start",
        [
            Event(),
            Event(
                global_step=1,
                global_batch=1,
                steps_per_epoch=1,
                batches_per_step=1,
                invocations_per_step=1,
            ),
        ],
    )
    def test_init(self, type_first, start):
        lifecycle = _get_event_lifecycle(type_first=type_first, start=start)
        assert lifecycle.type_first == type_first
        assert lifecycle.steps_per_epoch == start.steps_per_epoch
        assert lifecycle.batches_per_step == start.batches_per_step
        assert lifecycle.invocations_per_step == start.invocations_per_step
        assert lifecycle.global_step == start.global_step
        assert lifecycle.global_batch == start.global_batch

    @pytest.mark.parametrize(
        "type_, expected_func_name",
        [
            (EventType.BATCH_START, "batch_start_events"),
            (EventType.LOSS_CALCULATED, "loss_calculated_events"),
            (EventType.OPTIM_PRE_STEP, "optim_pre_step_events"),
            (EventType.OPTIM_POST_STEP, "optim_post_step_events"),
            (EventType.BATCH_END, "batch_end_events"),
        ],
    )
    def test_events_from_type_valid(self, type_, expected_func_name):
        lifecycle = _get_event_lifecycle()
        events, func_name = lifecycle.events_from_type(type_)

        assert events == []
        assert func_name == expected_func_name

    def test_events_from_type_raises_value_error(self):
        lifecycle = _get_event_lifecycle()
        with pytest.raises(ValueError):
            lifecycle.events_from_type("invalid")

    @pytest.mark.parametrize(
        "kwargs, increment ,expected",
        [
            ({"batches_per_step": None}, False, True),
            ({"batches_per_step": 1}, False, True),
            ({"batches_per_step": 3, "batch_count": 5}, False, True),
            ({"batches_per_step": 4, "batch_count": 7}, False, True),
            ({"batches_per_step": 4, "batch_count": 9}, False, False),
            ({"batches_per_step": 4, "batch_count": 9}, True, False),
            ({"batches_per_step": 4, "batch_count": 11}, True, True),
        ],
    )
    def test_check_step_batches_count(self, kwargs, increment, expected):
        lifecycle = _get_event_lifecycle()

        for key, value in kwargs.items():
            setattr(lifecycle, key, value)

        actual = lifecycle.check_step_batches_count(increment=increment)

        if increment:
            if not expected:
                assert lifecycle.batch_count == kwargs["batch_count"] + 1
            else:
                assert lifecycle.batch_count == 0

        assert actual == expected

    @pytest.mark.parametrize(
        "kwargs, increment ,expected",
        [
            ({"invocations_per_step": None}, False, True),
            ({"invocations_per_step": 1}, False, True),
            ({"invocations_per_step": 3, "step_count": 5}, False, True),
            ({"invocations_per_step": 4, "step_count": 7}, False, True),
            ({"invocations_per_step": 4, "step_count": 9}, False, False),
            ({"invocations_per_step": 4, "step_count": 9}, True, False),
            ({"invocations_per_step": 4, "step_count": 11}, True, True),
        ],
    )
    def test_check_step_invocations_count(self, kwargs, increment, expected):
        lifecycle = _get_event_lifecycle()

        for key, value in kwargs.items():
            setattr(lifecycle, key, value)

        actual = lifecycle.check_step_invocations_count(increment=increment)

        if increment:
            if not expected:
                assert lifecycle.step_count == kwargs["step_count"] + 1
            else:
                assert lifecycle.step_count == 0
        assert actual == expected


class TestWrappedOptimEventLifecycle:
    @pytest.mark.parametrize(
        "method_name",
        [
            "batch_start_events",
            "batch_end_events",
        ],
    )
    def test_batch_start_and_batch_end_events_are_invalid(self, method_name):
        # batch_start_events and batch_end_events must not be
        # called on WrappedOptimEventLifecycle explicitly
        # since they are auto-triggered when optim is wrapped

        lifecycle = _get_event_lifecycle(lifecycle_class=WrappedOptimEventLifecycle)
        with pytest.raises(ValueError, match="batch"):
            method = getattr(lifecycle, method_name)
            method()

    @pytest.mark.parametrize(
        "type_first",
        [
            EventType.BATCH_START,
            EventType.BATCH_END,
            EventType.OPTIM_PRE_STEP,
            EventType.OPTIM_POST_STEP,
        ],
    )
    def test_loss_calculated_events_with_invalid_first_event_type(self, type_first):
        # type_first must be EventType.LOSS_CALCULATED to get
        # loss_calculated_events on an
        # WrappedOptimEventLifecycle instance

        lifecycle = _get_event_lifecycle(
            type_first=type_first, lifecycle_class=WrappedOptimEventLifecycle
        )
        with pytest.raises(ValueError, match="loss calculated must"):
            lifecycle.loss_calculated_events()

    @pytest.mark.parametrize(
        "type_",
        [
            EventType.BATCH_START,
            EventType.OPTIM_PRE_STEP,
            EventType.BATCH_END,
        ],
    )
    def test_loss_calculated_events_with_invalid_event_type(self, type_):
        # type_ must be EventType.LOSS_CALCULATED or
        # EventType.OPITM_POST_STEP to get loss_calculated_events
        # on an WrappedOptimEventLifecycle instance

        lifecycle = _get_event_lifecycle(
            lifecycle_class=WrappedOptimEventLifecycle,
            type_first=EventType.LOSS_CALCULATED,
        )
        lifecycle.type_ = type_
        with pytest.raises(ValueError, match="loss calculated must"):
            lifecycle.loss_calculated_events()

    @pytest.mark.parametrize("check_step_batches_count_return", [True, False])
    def test_loss_calculated_events(self, monkeypatch, check_step_batches_count_return):
        lifecycle = _get_event_lifecycle(
            lifecycle_class=WrappedOptimEventLifecycle,
            type_first=EventType.LOSS_CALCULATED,
        )
        lifecycle.type_ = EventType.LOSS_CALCULATED

        def mock_check_step_batches_count(ret=True, *args, **kwargs):
            return ret

        monkeypatch.setattr(
            lifecycle,
            "check_step_batches_count",
            partial(mock_check_step_batches_count, ret=check_step_batches_count_return),
        )

        results = lifecycle.loss_calculated_events()

        assert isinstance(results, list) and len(results) >= 2
        assert results[0].type_ == EventType.BATCH_START
        assert results[1].type_ == EventType.LOSS_CALCULATED

        if not check_step_batches_count_return:
            assert len(results) == 3
            assert results[2].type_ == EventType.BATCH_END

    @pytest.mark.parametrize(
        "type_first, type_",
        [
            (EventType.OPTIM_PRE_STEP, EventType.BATCH_START),
            (EventType.OPTIM_PRE_STEP, EventType.LOSS_CALCULATED),
            (EventType.LOSS_CALCULATED, EventType.BATCH_START),
            (EventType.LOSS_CALCULATED, EventType.OPTIM_PRE_STEP),
            (EventType.LOSS_CALCULATED, EventType.OPTIM_POST_STEP),
        ],
    )
    def test_optim_pre_step_events_raises_value_error_with_invalid_event_invocation(
        self, type_first, type_
    ):
        # optim pre step  must be called before optim post step
        # and loss calculated must be called after loss calculation

        lifecycle = _get_event_lifecycle(
            lifecycle_class=WrappedOptimEventLifecycle, type_first=type_first
        )
        lifecycle.type_ = type_

        with pytest.raises(ValueError, match="optim pre step must"):
            lifecycle.optim_pre_step_events()

    @pytest.mark.parametrize(
        "type_first, type_, check_step_invocations_count_return",
        [
            (EventType.OPTIM_PRE_STEP, EventType.OPTIM_POST_STEP, False),
            (EventType.OPTIM_PRE_STEP, EventType.OPTIM_POST_STEP, True),
            (EventType.OPTIM_POST_STEP, EventType.OPTIM_POST_STEP, False),
        ],
    )
    def test_optim_pre_step_events(
        self, type_first, type_, check_step_invocations_count_return, monkeypatch
    ):
        lifecycle = _get_event_lifecycle(
            lifecycle_class=WrappedOptimEventLifecycle, type_first=type_first
        )
        lifecycle.type_ = type_

        def mock_check_step_invocations_count(ret=True, *args, **kwargs):
            return ret

        monkeypatch.setattr(
            lifecycle,
            "check_step_invocations_count",
            partial(
                mock_check_step_invocations_count,
                ret=check_step_invocations_count_return,
            ),
        )

        results = lifecycle.optim_pre_step_events()
        if type_first == EventType.OPTIM_PRE_STEP:
            assert len(results) >= 1
            assert results[0].type_ == EventType.BATCH_START

        if check_step_invocations_count_return:
            assert results[-1].type_ == EventType.OPTIM_PRE_STEP

    @pytest.mark.parametrize(
        "type_",
        [
            EventType.BATCH_START,
            EventType.BATCH_END,
            EventType.PRE_INIT,
        ],
    )
    def test_optim_post_step_events_raises_value_error_with_invalid_event_type(
        self, type_
    ):
        # optim post step must be called after optim pre step

        lifecycle = _get_event_lifecycle(lifecycle_class=WrappedOptimEventLifecycle)
        lifecycle.type_ = type_

        with pytest.raises(ValueError, match="optim post step must"):
            lifecycle.optim_post_step_events()

    @pytest.mark.parametrize(
        "type_, check_step_invocations_count_return",
        [
            (EventType.OPTIM_PRE_STEP, False),
            (EventType.OPTIM_PRE_STEP, True),
        ],
    )
    def test_optim_post_step_events(
        self, type_, monkeypatch, check_step_invocations_count_return
    ):
        lifecycle = _get_event_lifecycle(lifecycle_class=WrappedOptimEventLifecycle)
        lifecycle.type_ = type_

        def mock_check_step_invocations_count(ret=True, *args, **kwargs):
            return ret

        monkeypatch.setattr(
            lifecycle,
            "check_step_invocations_count",
            partial(
                mock_check_step_invocations_count,
                ret=check_step_invocations_count_return,
            ),
        )
        original_global_step = lifecycle.global_step

        results = lifecycle.optim_post_step_events()

        # type_ should be EventType.OPTIM_POST_STEP after
        # optim_post_step_events is called

        assert lifecycle.type_ == EventType.OPTIM_POST_STEP

        # check results

        if not check_step_invocations_count_return:
            assert lifecycle.global_step == original_global_step
            assert len(results) == 1
            assert results[0].type_ == EventType.BATCH_END
        else:
            assert lifecycle.global_step == original_global_step + 1
            assert len(results) == 2
            assert results[0].type_ == EventType.OPTIM_POST_STEP
            assert results[1].type_ == EventType.BATCH_END


class TestCallbackEventLifecycle:
    @pytest.mark.parametrize(
        "type_first, type_",
        [
            (EventType.BATCH_END, EventType.BATCH_START),
            (EventType.BATCH_START, EventType.BATCH_START),
            (EventType.BATCH_START, EventType.OPTIM_POST_STEP),
        ],
    )
    def test_batch_start_events_raises_value_error_with_invalid_event_invocation(
        self, type_first, type_
    ):
        # batch start must be called first for CallbacksEventLifecycle

        # batch start must be called after batch end for
        # CallbacksEventLifecycle

        lifecycle = _get_event_lifecycle(
            lifecycle_class=CallbacksEventLifecycle, type_first=type_first
        )
        lifecycle.type_ = type_

        with pytest.raises(ValueError, match="batch start must"):
            lifecycle.batch_start_events()

    @pytest.mark.parametrize(
        "type_first, type_",
        [
            (EventType.BATCH_START, EventType.BATCH_END),
        ],
    )
    def test_batch_start_events(self, type_first, type_):
        lifecycle = _get_event_lifecycle(
            lifecycle_class=CallbacksEventLifecycle, type_first=type_first
        )
        lifecycle.type_ = type_
        original_global_batch = lifecycle.global_batch
        results = lifecycle.batch_start_events()

        # type_ should be EventType.BATCH_START after
        # batch_start_events is called
        assert lifecycle.type_ == EventType.BATCH_START

        # global_batch should be incremented by 1
        assert lifecycle.global_batch == original_global_batch + 1

        assert len(results) == 1
        assert results[0].type_ == EventType.BATCH_START

    @pytest.mark.parametrize(
        "type_",
        [
            EventType.BATCH_END,
            EventType.OPTIM_PRE_STEP,
        ],
    )
    def test_loss_calculated_event_raises_value_error_with_invalid_event_type(
        self, type_
    ):
        # loss calculated must be called after batch start

        lifecycle = _get_event_lifecycle(lifecycle_class=CallbacksEventLifecycle)
        lifecycle.type_ = type_

        with pytest.raises(ValueError, match="loss calculated must"):
            lifecycle.loss_calculated_events()

    @pytest.mark.parametrize(
        "type_",
        [
            EventType.BATCH_START,
        ],
    )
    def test_loss_calculated_events(self, type_):
        lifecycle = _get_event_lifecycle(lifecycle_class=CallbacksEventLifecycle)
        lifecycle.type_ = type_

        results = lifecycle.loss_calculated_events()

        # type_ should be EventType.LOSS_CALCULATED after
        # loss_calculated_events is called
        assert lifecycle.type_ == EventType.LOSS_CALCULATED

        # check results
        assert len(results) == 1
        assert results[0].type_ == EventType.LOSS_CALCULATED

    @pytest.mark.parametrize(
        "type_",
        [
            EventType.BATCH_END,
            EventType.OPTIM_PRE_STEP,
            EventType.OPTIM_POST_STEP,
        ],
    )
    def test_optim_pre_step_events_raises_value_error_with_invalid_event_type(
        self, type_
    ):
        # optim pre step must be called after batch start or
        # loss calculation for CallbacksEventLifecycle

        lifecycle = _get_event_lifecycle(
            lifecycle_class=CallbacksEventLifecycle,
        )
        lifecycle.type_ = type_

        with pytest.raises(ValueError, match="optim pre step must"):
            lifecycle.optim_pre_step_events()

    @pytest.mark.parametrize(
        "type_, check_step_invocations_count_return",
        [
            (EventType.BATCH_START, False),
            (EventType.BATCH_START, True),
            (EventType.LOSS_CALCULATED, False),
            (EventType.LOSS_CALCULATED, True),
        ],
    )
    def test_optim_pre_step_events(
        self, type_, check_step_invocations_count_return, monkeypatch
    ):
        lifecycle = _get_event_lifecycle(
            lifecycle_class=CallbacksEventLifecycle,
        )
        lifecycle.type_ = type_

        def mock_check_step_invocations_count(ret=True, *args, **kwargs):
            return ret

        monkeypatch.setattr(
            lifecycle,
            "check_step_invocations_count",
            partial(
                mock_check_step_invocations_count,
                ret=check_step_invocations_count_return,
            ),
        )

        results = lifecycle.optim_pre_step_events()
        assert lifecycle.type_ == EventType.OPTIM_PRE_STEP

        if not check_step_invocations_count_return:
            assert len(results) == 0
        else:
            assert len(results) == 1
            assert results[0].type_ == EventType.OPTIM_PRE_STEP

    @pytest.mark.parametrize(
        "type_",
        [
            EventType.BATCH_START,
            EventType.BATCH_END,
            EventType.PRE_INIT,
            EventType.LOSS_CALCULATED,
            EventType.OPTIM_POST_STEP,
        ],
    )
    def test_optim_post_step_events_raises_value_error_with_invalid_event_type(
        self, type_
    ):
        # optim post step must be called after optim pre step

        lifecycle = _get_event_lifecycle(lifecycle_class=CallbacksEventLifecycle)
        lifecycle.type_ = type_

        with pytest.raises(ValueError, match="optim post step must"):
            lifecycle.optim_post_step_events()

    @pytest.mark.parametrize(
        "type_, check_step_invocations_count_return",
        [
            (EventType.OPTIM_PRE_STEP, False),
            (EventType.OPTIM_PRE_STEP, True),
        ],
    )
    def test_optim_post_step_events(
        self, type_, monkeypatch, check_step_invocations_count_return
    ):
        lifecycle = _get_event_lifecycle(lifecycle_class=CallbacksEventLifecycle)
        lifecycle.type_ = type_

        def mock_check_step_invocations_count(ret=True, *args, **kwargs):
            return ret

        monkeypatch.setattr(
            lifecycle,
            "check_step_invocations_count",
            partial(
                mock_check_step_invocations_count,
                ret=check_step_invocations_count_return,
            ),
        )
        original_global_step = lifecycle.global_step

        results = lifecycle.optim_post_step_events()

        # type_ should be EventType.OPTIM_POST_STEP after
        # optim_post_step_events is called

        assert lifecycle.type_ == EventType.OPTIM_POST_STEP

        # check results

        if not check_step_invocations_count_return:
            assert len(results) == 0
            assert lifecycle.global_batch == original_global_step
        else:
            assert lifecycle.global_step == original_global_step + 1
            assert len(results) == 1
            assert results[0].type_ == EventType.OPTIM_POST_STEP

    @pytest.mark.parametrize(
        "type_",
        [
            EventType.BATCH_END,
            EventType.OPTIM_PRE_STEP,
            EventType.PRE_INIT,
        ],
    )
    def test_batch_end_events_raises_value_error_with_invalid_event_type(self, type_):
        # batch end must be called after batch start or optim post step
        #  or loss calculation for CallbacksEventLifecycle

        lifecycle = _get_event_lifecycle(lifecycle_class=CallbacksEventLifecycle)
        lifecycle.type_ = type_

        with pytest.raises(ValueError, match="batch end must"):
            lifecycle.batch_end_events()

    @pytest.mark.parametrize(
        "type_",
        [
            EventType.OPTIM_POST_STEP,
            EventType.LOSS_CALCULATED,
            EventType.BATCH_START,
        ],
    )
    def test_batch_end_events(self, type_):
        lifecycle = _get_event_lifecycle(lifecycle_class=CallbacksEventLifecycle)
        lifecycle.type_ = type_

        results = lifecycle.batch_end_events()

        # type_ should be EventType.BATCH_END after
        # batch_end_events is called
        assert lifecycle.type_ == EventType.BATCH_END

        # check results
        assert len(results) == 1
        assert results[0].type_ == EventType.BATCH_END
