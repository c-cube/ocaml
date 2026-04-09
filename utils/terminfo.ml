(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Gallium, INRIA Paris                  *)
(*                                                                        *)
(*   Copyright 2017 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

open Printf

external terminfo_rows_fd: int -> int = "caml_stdlib_terminfo_rows"
(* Extract the raw fd from a new-style out_channel (returns -1 if not fd-backed). *)
external channel_fd_internal: out_channel -> int = "caml_stdlib_channel_fd"
let isatty oc = out_channel_isatty oc
let terminfo_rows oc =
  let fd = channel_fd_internal oc in
  if fd < 0 then -1 else terminfo_rows_fd fd

type status =
  | Uninitialised
  | Bad_term
  | Good_term

let setup oc =
  let term = try Sys.getenv "TERM" with Not_found -> "" in
  (* Same heuristics as in Misc.Color.should_enable_color *)
  if term <> "" && term <> "dumb" && isatty oc
  then Good_term
  else Bad_term

let num_lines oc =
  let rows = terminfo_rows oc in
  if rows > 0 then rows else 24
    (* 24 is a reasonable default for an ANSI-style terminal *)

let backup oc n =
  if n >= 1 then fprintf oc "\027[%dA%!" n

let resume oc n =
  if n >= 1 then fprintf oc "\027[%dB%!" n

let standout oc b =
  output_string oc (if b then "\027[4m" else "\027[0m"); flush oc
