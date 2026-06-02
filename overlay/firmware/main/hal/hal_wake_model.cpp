// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 @malonestar

/*
 * microWakeWord model selection — NVS-backed accessors.
 *
 * Used by the on-device Wake Word setup page (WakeWordSetupWorker) and read by
 * MicroWakeWord at construction so the selection takes effect on the next boot
 * (reboot-to-switch). Stored in NVS namespace "mww", key "model"; valid values
 * are the keys in MicroWakeWord::kModels ("stackchan" | "frank" | "m5").
 *
 * Part of the stackchan-mww add-on kit (not in the stock factory firmware).
 */
#include "hal.h"
#include <settings.h>

std::string Hal::getWakeModel()
{
    Settings s("mww", false);
    return s.GetString("model", "m5");
}

void Hal::setWakeModel(const std::string& v)
{
    Settings s("mww", true);
    s.SetString("model", v);
}
