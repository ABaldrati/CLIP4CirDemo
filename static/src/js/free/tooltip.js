import { getjQuery, onDOMContentLoaded } from '../mdb/util/index';
import EventHandler from '../mdb/dom/event-handler';
import BSTooltip from '../bootstrap/mdb-prefix/tooltip';
import SelectorEngine from '../mdb/dom/selector-engine';

/**
 * ------------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------------
 */

const NAME = 'tooltip';
const DATA_KEY = `mdb.${NAME}`;
const EVENT_KEY = `.${DATA_KEY}`;

const EVENT_HIDE_BS = 'hide.bs.tooltip';
const EVENT_HIDDEN_BS = 'hidden.bs.tooltip';
const EVENT_SHOW_BS = 'show.bs.tooltip';
const EVENT_SHOWN_BS = 'shown.bs.tooltip';
const EVENT_INSERTED_BS = 'inserted.bs.tooltip';

const EVENT_HIDE = `hide${EVENT_KEY}`;
const EVENT_HIDDEN = `hidden${EVENT_KEY}`;
const EVENT_SHOW = `show${EVENT_KEY}`;
const EVENT_SHOWN = `shown${EVENT_KEY}`;
const EVENT_INSERTED = `inserted${EVENT_KEY}`;

const SELECTOR_DATA_TOGGLE = '[data-mdb-toggle="tooltip"]';

class Tooltip extends BSTooltip {
  constructor(element, data) {
    super(element, data);

    this._init();
  }

  dispose() {
    EventHandler.off(this._element, EVENT_SHOW_BS);
    EventHandler.off(this._element, EVENT_SHOWN_BS);
    EventHandler.off(this._element, EVENT_HIDE_BS);
    EventHandler.off(this._element, EVENT_HIDDEN_BS);
    EventHandler.off(this._element, EVENT_INSERTED_BS);

    super.dispose();
  }

  // Getters
  static get NAME() {
    return NAME;
  }

  // Private
  _init() {
    this._bindShowEvent();
    this._bindShownEvent();
    this._bindHideEvent();
    this._bindHiddenEvent();
    this._bindHidePreventedEvent();
  }

  _bindShowEvent() {
    EventHandler.on(this.element, EVENT_SHOW_BS, () => {
      EventHandler.trigger(this.element, EVENT_SHOW);
    });
  }

  _bindShownEvent() {
    EventHandler.on(this.element, EVENT_SHOWN_BS, () => {
      EventHandler.trigger(this.element, EVENT_SHOWN);
    });
  }

  _bindHideEvent() {
    EventHandler.on(this.element, EVENT_HIDE_BS, () => {
      EventHandler.trigger(this.element, EVENT_HIDE);
    });
  }

  _bindHiddenEvent() {
    EventHandler.on(this.element, EVENT_HIDDEN_BS, () => {
      EventHandler.trigger(this.element, EVENT_HIDDEN);
    });
  }

  _bindHidePreventedEvent() {
    EventHandler.on(this.element, EVENT_INSERTED_BS, () => {
      EventHandler.trigger(this.element, EVENT_INSERTED);
    });
  }
}

/**
 * ------------------------------------------------------------------------
 * Data Api implementation - auto initialization
 * ------------------------------------------------------------------------
 */

SelectorEngine.find(SELECTOR_DATA_TOGGLE).forEach((el) => {
  let instance = Tooltip.getInstance(el);
  if (!instance) {
    instance = new Tooltip(el);
  }
});

/**
 * ------------------------------------------------------------------------
 * jQuery
 * ------------------------------------------------------------------------
 * add .rating to jQuery only if jQuery is present
 */

onDOMContentLoaded(() => {
  const $ = getjQuery();

  if ($) {
    const JQUERY_NO_CONFLICT = $.fn[NAME];
    $.fn[NAME] = Tooltip.jQueryInterface;
    $.fn[NAME].Constructor = Tooltip;
    $.fn[NAME].noConflict = () => {
      $.fn[NAME] = JQUERY_NO_CONFLICT;
      return Tooltip.jQueryInterface;
    };
  }
});

export default Tooltip;
