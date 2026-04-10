(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* Exceptions *)

external register_named_value : string -> 'a -> unit
                              = "caml_register_named_value"

let () =
  (* for runtime/fail_nat.c *)
  register_named_value "Pervasives.array_bound_error"
    (Invalid_argument "index out of bounds")

external raise : exn -> 'a = "%raise"
external raise_notrace : exn -> 'a = "%raise_notrace"

let failwith s = raise(Failure s)
let invalid_arg s = raise(Invalid_argument s)

exception Exit
exception Match_failure = Match_failure
exception Assert_failure = Assert_failure
exception Invalid_argument = Invalid_argument
exception Failure = Failure
exception Not_found = Not_found
exception Out_of_memory = Out_of_memory
exception Stack_overflow = Stack_overflow
exception Sys_error = Sys_error
exception End_of_file = End_of_file
exception Division_by_zero = Division_by_zero
exception Sys_blocked_io = Sys_blocked_io
exception Undefined_recursive_module = Undefined_recursive_module

(* Composition operators *)

external ( |> ) : 'a -> ('a -> 'b) -> 'b = "%revapply"
external ( @@ ) : ('a -> 'b) -> 'a -> 'b = "%apply"

(* Debugging *)

external __LOC__ : string = "%loc_LOC"
external __FILE__ : string = "%loc_FILE"
external __LINE__ : int = "%loc_LINE"
external __MODULE__ : string = "%loc_MODULE"
external __POS__ : string * int * int * int = "%loc_POS"
external __FUNCTION__ : string = "%loc_FUNCTION"

external __LOC_OF__ : 'a -> string * 'a = "%loc_LOC"
external __LINE_OF__ : 'a -> int * 'a = "%loc_LINE"
external __POS_OF__ : 'a -> (string * int * int * int) * 'a = "%loc_POS"

(* Comparisons *)

external ( = ) : 'a -> 'a -> bool = "%equal"
external ( <> ) : 'a -> 'a -> bool = "%notequal"
external ( < ) : 'a -> 'a -> bool = "%lessthan"
external ( > ) : 'a -> 'a -> bool = "%greaterthan"
external ( <= ) : 'a -> 'a -> bool = "%lessequal"
external ( >= ) : 'a -> 'a -> bool = "%greaterequal"
external compare : 'a -> 'a -> int = "%compare"

let min x y = if x <= y then x else y
let max x y = if x >= y then x else y

external ( == ) : 'a -> 'a -> bool = "%eq"
external ( != ) : 'a -> 'a -> bool = "%noteq"

(* Boolean operations *)

external not : bool -> bool = "%boolnot"
external ( && ) : bool -> bool -> bool = "%sequand"
external ( || ) : bool -> bool -> bool = "%sequor"

(* Integer operations *)

external ( ~- ) : int -> int = "%negint"
external ( ~+ ) : int -> int = "%identity"
external succ : int -> int = "%succint"
external pred : int -> int = "%predint"
external ( + ) : int -> int -> int = "%addint"
external ( - ) : int -> int -> int = "%subint"
external ( * ) : int -> int -> int = "%mulint"
external ( / ) : int -> int -> int = "%divint"
external ( mod ) : int -> int -> int = "%modint"

let abs x = if x >= 0 then x else -x

external ( land ) : int -> int -> int = "%andint"
external ( lor ) : int -> int -> int = "%orint"
external ( lxor ) : int -> int -> int = "%xorint"

let lnot x = x lxor (-1)

external ( lsl ) : int -> int -> int = "%lslint"
external ( lsr ) : int -> int -> int = "%lsrint"
external ( asr ) : int -> int -> int = "%asrint"

let max_int = (-1) lsr 1
let min_int = max_int + 1

(* Floating-point operations *)

external ( ~-. ) : float -> float = "%negfloat"
external ( ~+. ) : float -> float = "%identity"
external ( +. ) : float -> float -> float = "%addfloat"
external ( -. ) : float -> float -> float = "%subfloat"
external ( *. ) : float -> float -> float = "%mulfloat"
external ( /. ) : float -> float -> float = "%divfloat"
external ( ** ) : float -> float -> float = "caml_power_float" "pow"
  [@@unboxed] [@@noalloc]
external exp : float -> float = "caml_exp_float" "exp" [@@unboxed] [@@noalloc]
external expm1 : float -> float = "caml_expm1_float" "caml_expm1"
  [@@unboxed] [@@noalloc]
external acos : float -> float = "caml_acos_float" "acos"
  [@@unboxed] [@@noalloc]
external asin : float -> float = "caml_asin_float" "asin"
  [@@unboxed] [@@noalloc]
external atan : float -> float = "caml_atan_float" "atan"
  [@@unboxed] [@@noalloc]
external atan2 : float -> float -> float = "caml_atan2_float" "atan2"
  [@@unboxed] [@@noalloc]
external hypot : float -> float -> float
               = "caml_hypot_float" "caml_hypot" [@@unboxed] [@@noalloc]
external cos : float -> float = "caml_cos_float" "cos" [@@unboxed] [@@noalloc]
external cosh : float -> float = "caml_cosh_float" "cosh"
  [@@unboxed] [@@noalloc]
external acosh : float -> float = "caml_acosh_float" "caml_acosh"
  [@@unboxed] [@@noalloc]
external log : float -> float = "caml_log_float" "log" [@@unboxed] [@@noalloc]
external log10 : float -> float = "caml_log10_float" "log10"
  [@@unboxed] [@@noalloc]
external log1p : float -> float = "caml_log1p_float" "caml_log1p"
  [@@unboxed] [@@noalloc]
external sin : float -> float = "caml_sin_float" "sin" [@@unboxed] [@@noalloc]
external sinh : float -> float = "caml_sinh_float" "sinh"
  [@@unboxed] [@@noalloc]
external asinh : float -> float = "caml_asinh_float" "caml_asinh"
  [@@unboxed] [@@noalloc]
external sqrt : float -> float = "caml_sqrt_float" "sqrt"
  [@@unboxed] [@@noalloc]
external tan : float -> float = "caml_tan_float" "tan" [@@unboxed] [@@noalloc]
external tanh : float -> float = "caml_tanh_float" "tanh"
  [@@unboxed] [@@noalloc]
external atanh : float -> float = "caml_atanh_float" "caml_atanh"
  [@@unboxed] [@@noalloc]
external ceil : float -> float = "caml_ceil_float" "ceil"
  [@@unboxed] [@@noalloc]
external floor : float -> float = "caml_floor_float" "floor"
  [@@unboxed] [@@noalloc]
external abs_float : float -> float = "%absfloat"
external copysign : float -> float -> float
                  = "caml_copysign_float" "caml_copysign"
                  [@@unboxed] [@@noalloc]
external mod_float : float -> float -> float = "caml_fmod_float" "fmod"
  [@@unboxed] [@@noalloc]
external frexp : float -> float * int = "caml_frexp_float"
external ldexp : (float [@unboxed]) -> (int [@untagged]) -> (float [@unboxed]) =
  "caml_ldexp_float" "caml_ldexp_float_unboxed" [@@noalloc]
external modf : float -> float * float = "caml_modf_float"
external float : int -> float = "%floatofint"
external float_of_int : int -> float = "%floatofint"
external truncate : float -> int = "%intoffloat"
external int_of_float : float -> int = "%intoffloat"
external float_of_bits : int64 -> float
  = "caml_int64_float_of_bits" "caml_int64_float_of_bits_unboxed"
  [@@unboxed] [@@noalloc]
let infinity =
  float_of_bits 0x7F_F0_00_00_00_00_00_00L
let neg_infinity =
  float_of_bits 0xFF_F0_00_00_00_00_00_00L
let nan =
  float_of_bits 0x7F_F8_00_00_00_00_00_01L
let max_float =
  float_of_bits 0x7F_EF_FF_FF_FF_FF_FF_FFL
let min_float =
  float_of_bits 0x00_10_00_00_00_00_00_00L
let epsilon_float =
  float_of_bits 0x3C_B0_00_00_00_00_00_00L

type fpclass =
    FP_normal
  | FP_subnormal
  | FP_zero
  | FP_infinite
  | FP_nan
external classify_float : (float [@unboxed]) -> fpclass =
  "caml_classify_float" "caml_classify_float_unboxed" [@@noalloc]

(* String and byte sequence operations -- more in modules String and Bytes *)

external string_length : string -> int = "%string_length"
external bytes_length : bytes -> int = "%bytes_length"
external bytes_create : int -> bytes = "caml_create_bytes"
external string_blit : string -> int -> bytes -> int -> int -> unit
                     = "caml_blit_string" [@@noalloc]
external bytes_blit : bytes -> int -> bytes -> int -> int -> unit
                        = "caml_blit_bytes" [@@noalloc]
external bytes_unsafe_to_string : bytes -> string = "%bytes_to_string"
external bytes_unsafe_of_string : string -> bytes = "%bytes_to_string"

let ( ^ ) s1 s2 =
  let l1 = string_length s1 and l2 = string_length s2 in
  let s = bytes_create (l1 + l2) in
  string_blit s1 0 s 0 l1;
  string_blit s2 0 s l1 l2;
  bytes_unsafe_to_string s

(* Character operations -- more in module Char *)

external int_of_char : char -> int = "%identity"
external unsafe_char_of_int : int -> char = "%identity"
let char_of_int n =
  if n < 0 || n > 255 then invalid_arg "char_of_int" else unsafe_char_of_int n

(* Unit operations *)

external ignore : 'a -> unit = "%ignore"

(* Pair operations *)

external fst : 'a * 'b -> 'a = "%field0"
external snd : 'a * 'b -> 'b = "%field1"

(* References *)

type 'a ref = { mutable contents : 'a }
external ref : 'a -> 'a ref = "%makemutable"
external ( ! ) : 'a ref -> 'a = "%field0"
external ( := ) : 'a ref -> 'a -> unit = "%setfield0"
external incr : int ref -> unit = "%incr"
external decr : int ref -> unit = "%decr"

(* Result type *)

type ('a,'b) result = Ok of 'a | Error of 'b

(* String conversion functions *)

external format_int : string -> int -> string = "caml_format_int"
external format_float : string -> float -> string = "caml_format_float"

let string_of_bool b =
  if b then "true" else "false"
let bool_of_string = function
  | "true" -> true
  | "false" -> false
  | _ -> invalid_arg "bool_of_string"

let bool_of_string_opt = function
  | "true" -> Some true
  | "false" -> Some false
  | _ -> None

let string_of_int n =
  format_int "%d" n

external int_of_string : string -> int = "caml_int_of_string"

let int_of_string_opt s =
  (* Trashes current backtrace *)
  try Some (int_of_string s)
  with Failure _ -> None

external string_get : string -> int -> char = "%string_safe_get"

let valid_float_lexem s =
  let l = string_length s in
  let rec loop i =
    if i >= l then s ^ "." else
    match string_get s i with
    | '0' .. '9' | '-' -> loop (i + 1)
    | _ -> s
  in
  loop 0

let string_of_float f = valid_float_lexem (format_float "%.12g" f)

external float_of_string : string -> float = "caml_float_of_string"

let float_of_string_opt s =
  (* Trashes current backtrace *)
  try Some (float_of_string s)
  with Failure _ -> None

(* List operations -- more in module List *)

let[@tail_mod_cons] rec ( @ ) l1 l2 =
  match l1 with
  | [] -> l2
  | h1 :: [] -> h1 :: l2
  | h1 :: h2 :: [] -> h1 :: h2 :: l2
  | h1 :: h2 :: h3 :: tl -> h1 :: h2 :: h3 :: (tl @ l2)

(* I/O operations *)

(* ---- Bigarray primitives (local, before Bigarray module is available) ----

   The Bigarray module is compiled after stdlib, so we cannot reference
   Bigarray.Array1.t here.  Instead we define an abstract bigstring type
   and use the caml_ba_* C primitives directly.  The runtime does not
   check the OCaml type — it reads kind/layout from the bigarray header. *)

(* An abstract type wrapping a char C-layout 1-dimensional bigarray.
   The runtime representation is identical to Bigarray.Array1.t; we just
   cannot name that type here. *)
type bigstring

(* Create a fresh char/c_layout 1-d bigarray of the given length.
   kind=Char=12, layout=C_layout=0 per bigarray.h *)
external create_bigstring : int -> bigstring = "caml_ba_create_char_c1"

(* We use dedicated C stubs rather than the %caml_ba_* compiler primitives,
   because the compiler primitives require the Bigarray type to be statically
   known (otherwise cmmgen hits assert false for Pbigarray_unknown). *)
external ba_dim : bigstring -> int = "caml_bigstring_dim"
external ba_unsafe_get : bigstring -> int -> char = "caml_bigstring_unsafe_get"
external ba_unsafe_set : bigstring -> int -> char -> unit
  = "caml_bigstring_unsafe_set"

(* ---- Int64 arithmetic primitives (local) ---- *)

external int64_add : int64 -> int64 -> int64 = "%int64_add"
external int64_sub : int64 -> int64 -> int64 = "%int64_sub"
external int64_of_int : int -> int64 = "%int64_of_int"
external int64_to_int : int64 -> int = "%int64_to_int"

(* ---- Low-level I/O externals ---- *)

type open_flag =
    Open_rdonly | Open_wronly | Open_append
  | Open_creat | Open_trunc | Open_excl
  | Open_binary | Open_text | Open_nonblock

external open_desc : string -> open_flag list -> int -> int = "caml_sys_open"
external close_desc : int -> unit = "caml_sys_close"

external raw_read : int -> int -> bigstring -> int -> int -> int
  = "caml_stdlib_read"
external raw_write : int -> int -> bigstring -> int -> int -> int
  = "caml_stdlib_write"

external blit_bigstring_to_bytes :
  bigstring -> int -> bytes -> int -> int -> unit
  = "caml_blit_bigstring_to_bytes"
external blit_bytes_to_bigstring :
  bytes -> int -> bigstring -> int -> int -> unit
  = "caml_blit_bytes_to_bigstring"
external blit_string_to_bigstring :
  string -> int -> bigstring -> int -> int -> unit
  = "caml_blit_string_to_bigstring" [@@warning "-32"]

external raw_lseek : int -> int64 -> int -> int64 = "caml_stdlib_lseek"
external raw_isatty : int -> bool = "caml_stdlib_isatty" [@@warning "-32"]
external raw_set_binary_mode : int -> bool -> unit
  = "caml_stdlib_set_binary_mode"

(* ---- Marshal primitives (for output_value / input_value) ----
   Marshal module is not available yet; we use the C primitives directly. *)

external marshal_to_bytes : 'a -> unit list -> bytes
  = "caml_output_value_to_bytes"
external marshal_from_bytes_unsafe : bytes -> int -> 'a
  = "caml_input_value_from_bytes"
external marshal_data_size_unsafe : bytes -> int -> int
  = "caml_marshal_data_size"
let marshal_header_size = 16

(* ---- Local helpers ---- *)

let rec list_mem_eq x = function
  | [] -> false
  | y :: rest -> x = y || list_mem_eq x rest

(* ---- Constants ---- *)

let io_buffer_size = 65536

(* POSIX lseek whence constants *)
let seek_set_ = 0
let seek_cur_ = 1
let seek_end_ = 2

(* ---- Internal concurrency primitives (before Mutex/Atomic are available) ---- *)

module Mutex_ = struct
  type t
  external create : unit -> t = "caml_ml_mutex_new"
  external lock : t -> unit = "caml_ml_mutex_lock"
  external unlock : t -> unit = "caml_ml_mutex_unlock"
end

module Atomic_ = struct
  type 'a t
  external make : 'a -> 'a t = "%makemutable"
  external get : 'a t -> 'a = "%atomic_load"
  external compare_and_set : 'a t -> 'a -> 'a -> bool = "%atomic_cas"
end

external domain_self_id_ : unit -> int = "caml_ml_domain_id" [@@noalloc]

(* ---- Channel buffer type ---- *)

type chan_buffer = {
  bs: bigstring;
  mutable off: int;  (* start of valid data *)
  mutable len: int;  (* number of valid bytes from off *)
}

let make_chan_buffer () =
  { bs = create_bigstring io_buffer_size; off = 0; len = 0 }

(* ---- Fd state (for fd-backed channels) ---- *)

[@@@warning "-69"]
type fd_state = {
  fd: int;
  flags: int;  (* 0 on Unix; CHANNEL_FLAG_FROM_SOCKET on Windows *)
  mutable binary: bool;
  mutable buffered: bool;
  mutable name: string option;
  mutex: Mutex_.t;
  mutable lock_holder: int;  (* domain id holding the lock, or -1 *)
}
[@@@warning "+69"]

(* Re-entrant lock for fd out_channels. If the current domain already holds
   the lock (signal handler re-entering), skip the lock. *)
let[@inline] with_fd_lock (st : fd_state) f =
  let self = domain_self_id_ () in
  if st.lock_holder = self then f ()
  else begin
    Mutex_.lock st.mutex;
    st.lock_holder <- self;
    match f () with
    | x -> st.lock_holder <- -1; Mutex_.unlock st.mutex; x
    | exception e -> st.lock_holder <- -1; Mutex_.unlock st.mutex; raise e
  end

(* ---- Output channel ---- *)

type 'st out_ops = {
  out_write: 'st -> bigstring -> int -> int -> int;
  out_flush: 'st -> unit;
  out_close: 'st -> unit;
  out_seek: ('st -> int64 -> unit) option;
  out_pos: ('st -> int64) option;
  out_length: ('st -> int64) option;
  out_set_binary: ('st -> bool -> unit) option;
  out_isatty: ('st -> bool) option;
  out_is_binary: ('st -> bool) option;
  out_get_fd: ('st -> int) option;
  out_set_buffered: 'st -> bool -> unit;
  out_is_buffered: 'st -> bool;
}

(* Polymorphic equality (=) on channels:
   - Same channel (physically equal): the runtime's == fast path returns true
     immediately without inspecting any fields.
   - Distinct channels: [id] is the first field (an unboxed int), so (=)
     compares it first.  Since every channel gets a unique id, the comparison
     returns false at once — we never reach the vtable closures (which would
     raise "compare: functional value"). *)

let next_channel_id_ = Atomic_.make 0

let fresh_channel_id_ () =
  let rec loop () =
    let old = Atomic_.get next_channel_id_ in
    if Atomic_.compare_and_set next_channel_id_ old (old + 1) then old
    else loop ()
  in loop ()

type out_channel = Out_ch : {
  id: int;
  buf: chan_buffer;
  ops: 'st out_ops;
  st: 'st;
  mutable closed: bool;
} -> out_channel

(* ---- Input channel ---- *)

type 'st in_ops = {
  in_read: 'st -> chan_buffer -> unit;  (* refill: set off=0, update len *)
  in_close: 'st -> unit;
  in_seek: ('st -> int64 -> unit) option;
  in_pos: ('st -> int64) option;
  in_length: ('st -> int64) option;
  in_set_binary: ('st -> bool -> unit) option;
  in_isatty: ('st -> bool) option;
  in_is_binary: ('st -> bool) option;
  in_get_fd: ('st -> int) option;
}

type in_channel = In_ch : {
  id: int;
  buf: chan_buffer;
  ops: 'st in_ops;
  st: 'st;
  mutable closed: bool;
} -> in_channel

(* ---- Global tracking of fd-backed output channels for flush_all ---- *)

let all_out_channels : out_channel list Atomic_.t = Atomic_.make []

let register_out_channel (oc : out_channel) =
  let rec loop () =
    let old = Atomic_.get all_out_channels in
    if not (Atomic_.compare_and_set all_out_channels old (oc :: old))
    then loop ()
  in loop ()

let [@tail_mod_cons] rec filter_phys_neq v = function
  | [] -> []
  | x :: rest ->
    if x == v then filter_phys_neq v rest
    else x :: filter_phys_neq v rest

let unregister_out_channel (oc : out_channel) =
  let rec loop () =
    let old = Atomic_.get all_out_channels in
    let new_ = filter_phys_neq oc old in
    if not (Atomic_.compare_and_set all_out_channels old new_)
    then loop ()
  in loop ()

(* ---- Fd-backed vtables ---- *)

let fd_out_ops : fd_state out_ops = {
  out_write = (fun st bs ofs len ->
    with_fd_lock st (fun () -> raw_write st.fd st.flags bs ofs len));
  out_flush = (fun _st -> ());
  out_close = (fun st ->
    with_fd_lock st (fun () -> close_desc st.fd));
  out_seek = Some (fun st pos ->
    with_fd_lock st (fun () -> ignore (raw_lseek st.fd pos seek_set_)));
  out_pos = Some (fun st ->
    with_fd_lock st (fun () -> raw_lseek st.fd 0L seek_cur_));
  out_length = Some (fun st ->
    with_fd_lock st (fun () ->
      let cur = raw_lseek st.fd 0L seek_cur_ in
      let sz = raw_lseek st.fd 0L seek_end_ in
      ignore (raw_lseek st.fd cur seek_set_);
      sz));
  out_set_binary = Some (fun st bin ->
    with_fd_lock st (fun () ->
      raw_set_binary_mode st.fd bin;
      st.binary <- bin));
  out_isatty = Some (fun st -> raw_isatty st.fd);
  out_is_binary = Some (fun st -> st.binary);
  out_get_fd = Some (fun st -> st.fd);
  out_set_buffered = (fun st b -> st.buffered <- b);
  out_is_buffered = (fun st -> st.buffered);
}

let fd_in_ops : fd_state in_ops = {
  in_read = (fun st buf ->
    buf.off <- 0;
    let n = raw_read st.fd st.flags buf.bs 0 (ba_dim buf.bs) in
    buf.len <- n);
  in_close = (fun st -> close_desc st.fd);
  in_seek = Some (fun st pos ->
    ignore (raw_lseek st.fd pos seek_set_));
  in_pos = Some (fun st ->
    raw_lseek st.fd 0L seek_cur_);
  in_length = Some (fun st ->
    let cur = raw_lseek st.fd 0L seek_cur_ in
    let sz = raw_lseek st.fd 0L seek_end_ in
    ignore (raw_lseek st.fd cur seek_set_);
    sz);
  in_set_binary = Some (fun st bin ->
    raw_set_binary_mode st.fd bin;
    st.binary <- bin);
  in_isatty = Some (fun st -> raw_isatty st.fd);
  in_is_binary = Some (fun st -> st.binary);
  in_get_fd = Some (fun st -> st.fd);
}

(* ---- Channel constructors ---- *)

let make_fd_state fd flags binary name =
  { fd; flags; binary; buffered = true; name;
    mutex = Mutex_.create (); lock_holder = -1 }

let make_fd_out_channel fd flags binary name =
  let st = make_fd_state fd flags binary name in
  let oc = Out_ch { id = fresh_channel_id_ (); buf = make_chan_buffer ();
                    ops = fd_out_ops; st; closed = false } in
  register_out_channel oc;
  oc

let make_fd_in_channel fd flags binary name =
  In_ch { id = fresh_channel_id_ (); buf = make_chan_buffer (); ops = fd_in_ops;
          st = make_fd_state fd flags binary name; closed = false }

(* ---- Public constructors for fd-backed channels ---- *)

let open_descriptor_in fd =
  make_fd_in_channel fd 0 true None

let open_descriptor_out fd =
  make_fd_out_channel fd 0 true None

let in_channel_fd (ic : in_channel) : int =
  let (In_ch r) = ic in
  match r.ops.in_get_fd with
  | Some f -> f r.st
  | None -> invalid_arg "in_channel_fd: not a file-descriptor channel"

let out_channel_fd (oc : out_channel) : int =
  let (Out_ch r) = oc in
  match r.ops.out_get_fd with
  | Some f -> f r.st
  | None -> invalid_arg "out_channel_fd: not a file-descriptor channel"

(* ---- Standard channels ---- *)

let stdin  = make_fd_in_channel  0 0 false None
let stdout = make_fd_out_channel 1 0 false None
let stderr = make_fd_out_channel 2 0 false None

(* ==== Output functions ==== *)

(* Internal: flush buffer contents via out_write, handling partial writes.
   If the channel is closed (e.g. by a signal handler) during a write,
   discard remaining buffer data and return. *)
let flush_buf (Out_ch r) =
  while r.buf.len > 0 && not r.closed do
    match r.ops.out_write r.st r.buf.bs r.buf.off r.buf.len with
    | n ->
      r.buf.off <- r.buf.off + n;
      r.buf.len <- r.buf.len - n
    | exception Sys_error _ when r.closed ->
      (* Signal handler closed the channel; discard remaining data *)
      r.buf.len <- 0
  done;
  r.buf.off <- 0

let flush (oc : out_channel) =
  let (Out_ch r) = oc in
  if not r.closed then begin
    flush_buf oc;
    r.ops.out_flush r.st
  end

let flush_all () =
  let rec iter = function
    | [] -> ()
    | a :: l ->
      begin try
        flush a
      with Sys_error _ ->
        () (* ignore channels closed during a preceding flush *)
      end;
      iter l
  in
  iter (Atomic_.get all_out_channels)

let output_char (oc : out_channel) (c : char) =
  let (Out_ch r) = oc in
  if r.closed then raise (Sys_error "output_char: channel is closed");
  let cap = ba_dim r.buf.bs in
  if r.buf.off + r.buf.len >= cap then flush_buf oc;
  ba_unsafe_set r.buf.bs (r.buf.off + r.buf.len) c;
  r.buf.len <- r.buf.len + 1;
  if r.buf.off + r.buf.len >= cap || not (r.ops.out_is_buffered r.st)
  then flush_buf oc

let output_byte (oc : out_channel) (n : int) =
  output_char oc (unsafe_char_of_int (n land 0xFF))

let output (oc : out_channel) (s : bytes) (ofs : int) (len : int) =
  if ofs < 0 || len < 0 || ofs > bytes_length s - len
  then invalid_arg "output";
  let (Out_ch r) = oc in
  if r.closed then raise (Sys_error "output: channel is closed");
  let cap = ba_dim r.buf.bs in
  let i = ref ofs in
  let remaining = ref len in
  while !remaining > 0 do
    if r.buf.off + r.buf.len >= cap then flush_buf oc;
    let n = min !remaining (cap - r.buf.off - r.buf.len) in
    blit_bytes_to_bigstring s !i r.buf.bs (r.buf.off + r.buf.len) n;
    r.buf.len <- r.buf.len + n;
    i := !i + n;
    remaining := !remaining - n
  done;
  if r.buf.off + r.buf.len >= cap || not (r.ops.out_is_buffered r.st)
  then flush_buf oc

let output_substring (oc : out_channel) (s : string) (ofs : int) (len : int) =
  if ofs < 0 || len < 0 || ofs > string_length s - len
  then invalid_arg "output_substring"
  else output oc (bytes_unsafe_of_string s) ofs len

let output_bytes oc s = output oc s 0 (bytes_length s)
let output_string oc s = output_substring oc s 0 (string_length s)

let output_binary_int oc (n : int) =
  output_byte oc (n asr 24);
  output_byte oc (n asr 16);
  output_byte oc (n asr 8);
  output_byte oc n

let output_value oc v =
  let s = marshal_to_bytes v [] in
  output_bytes oc s

let seek_out (oc : out_channel) (pos : int) =
  let (Out_ch r) = oc in
  if r.closed then raise (Sys_error "seek_out: channel is closed");
  flush oc;
  match r.ops.out_seek with
  | None -> invalid_arg "seek_out: channel does not support seeking"
  | Some f -> f r.st (int64_of_int pos)

let pos_out (oc : out_channel) =
  let (Out_ch r) = oc in
  if r.closed then raise (Sys_error "pos_out: channel is closed");
  match r.ops.out_pos with
  | None -> invalid_arg "pos_out: channel does not support position"
  | Some f ->
    let raw = f r.st in
    int64_to_int (int64_add raw (int64_of_int r.buf.len))

let out_channel_length (oc : out_channel) =
  let (Out_ch r) = oc in
  if r.closed then raise (Sys_error "out_channel_length: channel is closed");
  match r.ops.out_length with
  | None ->
    invalid_arg "out_channel_length: channel does not support length"
  | Some f -> int64_to_int (f r.st)

let close_out_channel (oc : out_channel) =
  let (Out_ch r) = oc in
  if not r.closed then begin
    r.closed <- true;
    unregister_out_channel oc;
    (try flush_buf oc with _ -> ());
    r.ops.out_close r.st
  end

let close_out oc = flush oc; close_out_channel oc

let close_out_noerr oc =
  (try flush oc with _ -> ());
  (try close_out_channel oc with _ -> ())

let set_binary_mode_out (oc : out_channel) (bin : bool) =
  let (Out_ch r) = oc in
  if r.closed then raise (Sys_error "set_binary_mode_out: channel is closed");
  flush_buf oc;
  match r.ops.out_set_binary with
  | None -> ()
  | Some f -> f r.st bin

let out_channel_isatty (oc : out_channel) =
  let (Out_ch r) = oc in
  match r.ops.out_isatty with
  | None -> false
  | Some f -> f r.st

let out_channel_is_binary_mode (oc : out_channel) =
  let (Out_ch r) = oc in
  match r.ops.out_is_binary with
  | None -> false
  | Some f -> f r.st

let set_buffered_out (oc : out_channel) (b : bool) =
  let (Out_ch r) = oc in
  if r.closed then raise (Sys_error "set_buffered: channel is closed");
  if not b then flush_buf oc;
  r.ops.out_set_buffered r.st b

let is_buffered_out (oc : out_channel) : bool =
  let (Out_ch r) = oc in
  r.ops.out_is_buffered r.st

(* ---- open_out ---- *)

let open_out_gen mode perm name =
  let fd = open_desc name mode perm in
  let binary = list_mem_eq Open_binary mode in
  let oc = make_fd_out_channel fd 0 binary (Some name) in
  oc

let open_out name =
  open_out_gen [Open_wronly; Open_creat; Open_trunc; Open_text] 0o666 name

let open_out_bin name =
  open_out_gen [Open_wronly; Open_creat; Open_trunc; Open_binary] 0o666 name

(* ==== Input functions ==== *)

let input_char (ic : in_channel) =
  let (In_ch r) = ic in
  if r.closed then raise (Sys_error "input_char: channel is closed");
  if r.buf.len = 0 then begin
    r.ops.in_read r.st r.buf;
    if r.buf.len = 0 then raise End_of_file
  end;
  let c = ba_unsafe_get r.buf.bs r.buf.off in
  r.buf.off <- r.buf.off + 1;
  r.buf.len <- r.buf.len - 1;
  c

let input_byte (ic : in_channel) =
  int_of_char (input_char ic)

(* Internal: read into user's bytes buffer without bounds checking *)
let unsafe_input (ic : in_channel) (s : bytes) (ofs : int) (len : int) =
  let (In_ch r) = ic in
  if r.closed then raise (Sys_error "input: channel is closed");
  if r.buf.len = 0 then begin
    r.ops.in_read r.st r.buf;
    if r.buf.len = 0 then 0
    else begin
      let n = min r.buf.len len in
      blit_bigstring_to_bytes r.buf.bs r.buf.off s ofs n;
      r.buf.off <- r.buf.off + n;
      r.buf.len <- r.buf.len - n;
      n
    end
  end else begin
    let n = min r.buf.len len in
    blit_bigstring_to_bytes r.buf.bs r.buf.off s ofs n;
    r.buf.off <- r.buf.off + n;
    r.buf.len <- r.buf.len - n;
    n
  end

let input ic s ofs len =
  if ofs < 0 || len < 0 || ofs > bytes_length s - len
  then invalid_arg "input"
  else unsafe_input ic s ofs len

let rec unsafe_really_input ic s ofs len =
  if len <= 0 then () else begin
    let r = unsafe_input ic s ofs len in
    if r = 0
    then raise End_of_file
    else unsafe_really_input ic s (ofs + r) (len - r)
  end

let really_input ic s ofs len =
  if ofs < 0 || len < 0 || ofs > bytes_length s - len
  then invalid_arg "really_input"
  else unsafe_really_input ic s ofs len

let really_input_string ic len =
  let s = bytes_create len in
  really_input ic s 0 len;
  bytes_unsafe_to_string s

(* Copy a slice of bigstring into a fresh bytes *)
let bytes_of_bigstring_sub (bs : bigstring) (off : int) (len : int) : bytes =
  let b = bytes_create len in
  if len > 0 then blit_bigstring_to_bytes bs off b 0 len;
  b

(* Scan [buf] for '\n' in [buf.off .. buf.off+buf.len-1].
   Returns the index relative to [off], or -1 if not found.
   Takes all parameters explicitly to avoid closure allocation. *)
let scan_newline (buf : chan_buffer) : int =
  let bs = buf.bs in
  let off = buf.off in
  let limit = off + buf.len in
  let rec loop i =
    if i >= limit then -1
    else if ba_unsafe_get bs i = '\n' then i - off
    else loop (i + 1)
  in
  loop off

(* Consume [n] bytes from buffer, return them as bytes *)
let consume_buf (buf : chan_buffer) (n : int) : bytes =
  let b = bytes_of_bigstring_sub buf.bs buf.off n in
  buf.off <- buf.off + n;
  buf.len <- buf.len - n;
  b

(* Concatenate a reversed list of chunks into a single string *)
let concat_chunks_rev (chunks : bytes list) (total : int) : string =
  let result = bytes_create total in
  let pos = ref total in
  let rec copy = function
    | [] -> ()
    | c :: rest ->
      let n = bytes_length c in
      pos := !pos - n;
      bytes_blit c 0 result !pos n;
      copy rest
  in
  copy chunks;
  bytes_unsafe_to_string result

let input_line (ic : in_channel) =
  let (In_ch r) = ic in
  if r.closed then raise (Sys_error "input_line: channel is closed");
  (* Fast path: newline already in buffer *)
  let nl = scan_newline r.buf in
  if nl >= 0 then begin
    let line = consume_buf r.buf nl in
    r.buf.off <- r.buf.off + 1; (* skip '\n' *)
    r.buf.len <- r.buf.len - 1;
    bytes_unsafe_to_string line
  end else begin
    (* Slow path: accumulate chunks across refills *)
    let rec collect chunks total_len =
      if r.buf.len = 0 then begin
        r.ops.in_read r.st r.buf;
        if r.buf.len = 0 then begin
          (* EOF *)
          if total_len = 0 then raise End_of_file
          else concat_chunks_rev chunks total_len
        end else
          collect chunks total_len
      end else
        let nl = scan_newline r.buf in
        if nl >= 0 then begin
          let chunk = consume_buf r.buf nl in
          r.buf.off <- r.buf.off + 1;
          r.buf.len <- r.buf.len - 1;
          concat_chunks_rev (chunk :: chunks) (total_len + nl)
        end else begin
          let chunk = consume_buf r.buf r.buf.len in
          r.ops.in_read r.st r.buf;
          collect (chunk :: chunks) (total_len + bytes_length chunk)
        end
    in
    collect [] 0
  end

let input_binary_int ic =
  let b0 = input_byte ic in
  let b1 = input_byte ic in
  let b2 = input_byte ic in
  let b3 = input_byte ic in
  (* Big-endian 4-byte signed integer.
     We compute this portably (bytecode compat-32) by treating b0 as a
     signed byte: on 64-bit platforms this naturally sign-extends the
     result, matching the old C implementation. All constants are small. *)
  let s0 = if b0 land 0x80 <> 0 then b0 - 256 else b0 in
  (s0 lsl 24) lor (b1 lsl 16) lor (b2 lsl 8) lor b3

let input_value ic =
  let header = bytes_create marshal_header_size in
  really_input ic header 0 marshal_header_size;
  let data_size = marshal_data_size_unsafe header 0 in
  let buf = bytes_create (marshal_header_size + data_size) in
  bytes_blit header 0 buf 0 marshal_header_size;
  really_input ic buf marshal_header_size data_size;
  marshal_from_bytes_unsafe buf 0

let seek_in (ic : in_channel) (pos : int) =
  let (In_ch r) = ic in
  if r.closed then raise (Sys_error "seek_in: channel is closed");
  (* Discard buffered data *)
  r.buf.off <- 0;
  r.buf.len <- 0;
  match r.ops.in_seek with
  | None -> invalid_arg "seek_in: channel does not support seeking"
  | Some f -> f r.st (int64_of_int pos)

let pos_in (ic : in_channel) =
  let (In_ch r) = ic in
  if r.closed then raise (Sys_error "pos_in: channel is closed");
  match r.ops.in_pos with
  | None -> invalid_arg "pos_in: channel does not support position"
  | Some f ->
    (* Kernel position minus buffered unread data *)
    let raw = f r.st in
    int64_to_int (int64_sub raw (int64_of_int r.buf.len))

let in_channel_length (ic : in_channel) =
  let (In_ch r) = ic in
  if r.closed then raise (Sys_error "in_channel_length: channel is closed");
  match r.ops.in_length with
  | None ->
    invalid_arg "in_channel_length: channel does not support length"
  | Some f -> int64_to_int (f r.st)

let close_in (ic : in_channel) =
  let (In_ch r) = ic in
  if not r.closed then begin
    r.closed <- true;
    r.buf.off <- 0;
    r.buf.len <- 0;
    r.ops.in_close r.st
  end

let close_in_noerr ic = (try close_in ic with _ -> ())

let set_binary_mode_in (ic : in_channel) (bin : bool) =
  let (In_ch r) = ic in
  if r.closed then raise (Sys_error "set_binary_mode_in: channel is closed");
  match r.ops.in_set_binary with
  | None -> ()
  | Some f -> f r.st bin

let in_channel_isatty (ic : in_channel) =
  let (In_ch r) = ic in
  match r.ops.in_isatty with
  | None -> false
  | Some f -> f r.st

let in_channel_is_binary_mode (ic : in_channel) =
  let (In_ch r) = ic in
  match r.ops.in_is_binary with
  | None -> false
  | Some f -> f r.st

(* ---- open_in ---- *)

let open_in_gen mode perm name =
  let fd = open_desc name mode perm in
  let binary = list_mem_eq Open_binary mode in
  make_fd_in_channel fd 0 binary (Some name)

let open_in name =
  open_in_gen [Open_rdonly; Open_text] 0 name

let open_in_bin name =
  open_in_gen [Open_rdonly; Open_binary] 0 name

(* Output functions on standard output *)

let print_char c = output_char stdout c
let print_string s = output_string stdout s
let print_bytes s = output_bytes stdout s
let print_int i = output_string stdout (string_of_int i)
let print_float f = output_string stdout (string_of_float f)
let print_endline s =
  output_string stdout s; output_char stdout '\n'; flush stdout
let print_newline () = output_char stdout '\n'; flush stdout

(* Output functions on standard error *)

let prerr_char c = output_char stderr c
let prerr_string s = output_string stderr s
let prerr_bytes s = output_bytes stderr s
let prerr_int i = output_string stderr (string_of_int i)
let prerr_float f = output_string stderr (string_of_float f)
let prerr_endline s =
  output_string stderr s; output_char stderr '\n'; flush stderr
let prerr_newline () = output_char stderr '\n'; flush stderr

(* Input functions on standard input *)

let read_line () = flush stdout; input_line stdin
let read_int () = int_of_string(read_line())
let read_int_opt () = int_of_string_opt(read_line())
let read_float () = float_of_string(read_line())
let read_float_opt () = float_of_string_opt(read_line())

(* Operations on large files *)

module LargeFile = struct
  let seek_out (oc : out_channel) (pos : int64) =
    let (Out_ch r) = oc in
    if r.closed then raise (Sys_error "seek_out: channel is closed");
    flush oc;
    match r.ops.out_seek with
    | None -> invalid_arg "seek_out: channel does not support seeking"
    | Some f -> f r.st pos

  let pos_out (oc : out_channel) =
    let (Out_ch r) = oc in
    if r.closed then raise (Sys_error "pos_out: channel is closed");
    match r.ops.out_pos with
    | None -> invalid_arg "pos_out: channel does not support position"
    | Some f -> int64_add (f r.st) (int64_of_int r.buf.len)

  let out_channel_length (oc : out_channel) =
    let (Out_ch r) = oc in
    if r.closed then
      raise (Sys_error "out_channel_length: channel is closed");
    match r.ops.out_length with
    | None ->
      invalid_arg "out_channel_length: channel does not support length"
    | Some f -> f r.st

  let seek_in (ic : in_channel) (pos : int64) =
    let (In_ch r) = ic in
    if r.closed then raise (Sys_error "seek_in: channel is closed");
    r.buf.off <- 0;
    r.buf.len <- 0;
    match r.ops.in_seek with
    | None -> invalid_arg "seek_in: channel does not support seeking"
    | Some f -> f r.st pos

  let pos_in (ic : in_channel) =
    let (In_ch r) = ic in
    if r.closed then raise (Sys_error "pos_in: channel is closed");
    match r.ops.in_pos with
    | None -> invalid_arg "pos_in: channel does not support position"
    | Some f -> int64_sub (f r.st) (int64_of_int r.buf.len)

  let in_channel_length (ic : in_channel) =
    let (In_ch r) = ic in
    if r.closed then
      raise (Sys_error "in_channel_length: channel is closed");
    match r.ops.in_length with
    | None ->
      invalid_arg "in_channel_length: channel does not support length"
    | Some f -> f r.st
end

(* Formats *)

type ('a, 'b, 'c, 'd, 'e, 'f) format6
   = ('a, 'b, 'c, 'd, 'e, 'f) CamlinternalFormatBasics.format6
   = Format of ('a, 'b, 'c, 'd, 'e, 'f) CamlinternalFormatBasics.fmt
               * string

type ('a, 'b, 'c, 'd) format4 = ('a, 'b, 'c, 'c, 'c, 'd) format6

type ('a, 'b, 'c) format = ('a, 'b, 'c, 'c) format4

let string_of_format (Format (_fmt, str)) = str

external format_of_string :
 ('a, 'b, 'c, 'd, 'e, 'f) format6 ->
 ('a, 'b, 'c, 'd, 'e, 'f) format6 = "%identity"

let ( ^^ ) (Format (fmt1, str1)) (Format (fmt2, str2)) =
  Format (CamlinternalFormatBasics.concat_fmt fmt1 fmt2,
          str1 ^ "%," ^ str2)

(* Miscellaneous *)

external sys_exit : int -> 'a = "caml_sys_exit"

(* for at_exit *)
type 'a atomic_t
external atomic_make : 'a -> 'a atomic_t = "%makemutable"
external atomic_get : 'a atomic_t -> 'a = "%atomic_load"
external atomic_compare_and_set : 'a atomic_t -> 'a -> 'a -> bool
  = "%atomic_cas"

let exit_function = atomic_make flush_all

let rec at_exit f =
  (* MPR#7253, MPR#7796: make sure "f" is executed only once *)
  let f_yet_to_run = atomic_make true in
  let old_exit = atomic_get exit_function in
  let new_exit () =
    if atomic_compare_and_set f_yet_to_run true false then f () ;
    old_exit ()
  in
  let success = atomic_compare_and_set exit_function old_exit new_exit in
  if not success then at_exit f

let do_domain_local_at_exit = ref (fun () -> ())

let do_at_exit () =
  (!do_domain_local_at_exit) ();
  (atomic_get exit_function) ()

let exit retcode =
  do_at_exit ();
  sys_exit retcode

let _ = register_named_value "Pervasives.do_at_exit" do_at_exit

(*MODULE_ALIASES*)
module Arg            = Arg
module Array          = Array
module ArrayLabels    = ArrayLabels
module Atomic         = Atomic
module Bigarray       = Bigarray
module Bool           = Bool
module Buffer         = Buffer
module Bytes          = Bytes
module BytesLabels    = BytesLabels
module Callback       = Callback
module Char           = Char
module Complex        = Complex
module Condition      = Condition
module Digest         = Digest
module Domain         = Domain
module Dynarray       = Dynarray
module Pqueue         = Pqueue
module Effect         = Effect
module Either         = Either
module Ephemeron      = Ephemeron
module Filename       = Filename
module Float          = Float
module Format         = Format
module Fun            = Fun
module Gc             = Gc
module Hashtbl        = Hashtbl
module Iarray         = Iarray
module In_channel     = In_channel
module Int            = Int
module Int32          = Int32
module Int64          = Int64
module Lazy           = Lazy
module Lexing         = Lexing
module List           = List
module ListLabels     = ListLabels
module Map            = Map
module Marshal        = Marshal
module MoreLabels     = MoreLabels
module Mutex          = Mutex
module Nativeint      = Nativeint
module Obj            = Obj
module Oo             = Oo
module Option         = Option
module Out_channel    = Out_channel
module Pair           = Pair
module Parsing        = Parsing
module Printexc       = Printexc
module Printf         = Printf
module Queue          = Queue
module Random         = Random
module Result         = Result
module Repr           = Repr
module Scanf          = Scanf
module Semaphore      = Semaphore
module Seq            = Seq
module Set            = Set
module Stack          = Stack
module StdLabels      = StdLabels
module String         = String
module StringLabels   = StringLabels
module Sys            = Sys
module Type           = Type
module Uchar          = Uchar
module Unit           = Unit
module Weak           = Weak
