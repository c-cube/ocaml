/**************************************************************************/
/*                                                                        */
/*                                 OCaml                                  */
/*                                                                        */
/*             Simon Cruanes                                              */
/*                                                                        */
/*   Copyright 2026 Institut National de Recherche en Informatique et     */
/*     en Automatique.                                                    */
/*                                                                        */
/*   All rights reserved.  This file is distributed under the terms of    */
/*   the GNU Lesser General Public License version 2.1, with the          */
/*   special exception on linking described in the file LICENSE.          */
/*                                                                        */
/**************************************************************************/

#define CAML_INTERNALS

/* Low-level I/O stubs for the pure-OCaml channel implementation.
   All read/write operations use bigstring (Bigarray) buffers, which live
   outside the OCaml heap and are therefore safe to pass across blocking
   sections (GC cannot move them). */

#include <errno.h>
#include <unistd.h>
#include "caml/mlvalues.h"
#include "caml/memory.h"
#include "caml/signals.h"
#include "caml/bigarray.h"
#include "caml/osdeps.h"
#include "caml/io.h"  /* for file_offset, CHANNEL_FLAG_FROM_SOCKET */
#include "caml/alloc.h"
#include "caml/sys.h"

/* Read from fd directly into bigstring.
   GC-safe: bigstring data lives outside the OCaml heap.
   fd:int -> flags:int -> bigstring -> ofs:int -> len:int -> int */
CAMLprim value caml_stdlib_read(value vfd, value vflags,
                                value vba, value vofs, value vlen) {
  CAMLparam1(vba);
  char *buf = (char *)Caml_ba_data_val(vba) + Long_val(vofs);
  int n = caml_read_fd(Int_val(vfd), Int_val(vflags), buf, Int_val(vlen));
  if (n == -1) caml_sys_io_error(NO_ARG);
  CAMLreturn(Val_int(n));
}

/* Write from bigstring directly to fd.
   GC-safe: same reason.
   fd:int -> flags:int -> bigstring -> ofs:int -> len:int -> int */
CAMLprim value caml_stdlib_write(value vfd, value vflags,
                                 value vba, value vofs, value vlen) {
  CAMLparam1(vba);
  char *buf = (char *)Caml_ba_data_val(vba) + Long_val(vofs);
  int n = caml_write_fd(Int_val(vfd), Int_val(vflags), buf, Int_val(vlen));
  if (n == -1) caml_sys_io_error(NO_ARG);
  CAMLreturn(Val_int(n));
}

/* Blit bigstring -> bytes.
   Called while runtime is held, so OCaml bytes won't move.
   bigstring -> bs_ofs:int -> bytes -> bytes_ofs:int -> len:int -> unit */
CAMLprim value caml_blit_bigstring_to_bytes(value vba, value vba_ofs,
    value vbytes, value vbytes_ofs, value vlen) {
  memmove(&Byte(vbytes, Long_val(vbytes_ofs)),
          (char *)Caml_ba_data_val(vba) + Long_val(vba_ofs),
          Long_val(vlen));
  return Val_unit;
}

/* Blit bytes -> bigstring.
   bytes -> bytes_ofs:int -> bigstring -> bs_ofs:int -> len:int -> unit */
CAMLprim value caml_blit_bytes_to_bigstring(value vbytes, value vbytes_ofs,
    value vba, value vba_ofs, value vlen) {
  memmove((char *)Caml_ba_data_val(vba) + Long_val(vba_ofs),
          &Byte(vbytes, Long_val(vbytes_ofs)),
          Long_val(vlen));
  return Val_unit;
}

/* Blit string -> bigstring (string and bytes have the same representation).
   string -> str_ofs:int -> bigstring -> bs_ofs:int -> len:int -> unit */
CAMLprim value caml_blit_string_to_bigstring(value vstr, value vstr_ofs,
    value vba, value vba_ofs, value vlen) {
  char *dst = (char *)Caml_ba_data_val(vba) + Long_val(vba_ofs);
  char *src = (char *)vstr + Long_val(vstr_ofs);
  intnat len = Long_val(vlen);
  memmove(dst, src, len);
  return Val_unit;
}

/* Bootstrap aliases: boot/ocamlc still references these names via -use-prims.
   They are called via C_CALL5 (5 individual value args), NOT C_CALLN.
   So they must have the same signature as the real functions. */
CAMLprim value caml_stdlib_read_bc(value a, value b, value c, value d, value e) {
  return caml_stdlib_read(a, b, c, d, e);
}
CAMLprim value caml_stdlib_write_bc(value a, value b, value c, value d, value e) {
  return caml_stdlib_write(a, b, c, d, e);
}
CAMLprim value caml_blit_bigstring_to_bytes_bc(value a, value b, value c, value d, value e) {
  return caml_blit_bigstring_to_bytes(a, b, c, d, e);
}
CAMLprim value caml_blit_bytes_to_bigstring_bc(value a, value b, value c, value d, value e) {
  return caml_blit_bytes_to_bigstring(a, b, c, d, e);
}
CAMLprim value caml_blit_string_to_bigstring_bc(value a, value b, value c, value d, value e) {
  return caml_blit_string_to_bigstring(a, b, c, d, e);
}

/* lseek: fd:int -> offset:int64 -> whence:int -> int64 */
CAMLprim value caml_stdlib_lseek(value vfd, value vpos, value vwhence) {
  file_offset pos;
  caml_enter_blocking_section_no_pending();
  pos = lseek(Int_val(vfd), File_offset_val(vpos), Int_val(vwhence));
  caml_leave_blocking_section();
  if (pos == -1) caml_sys_io_error(NO_ARG);
  return Val_file_offset(pos);
}

/* isatty on raw fd: fd:int -> bool */
CAMLprim value caml_stdlib_isatty(value vfd) {
#ifdef _WIN32
  return Val_bool(caml_win32_isatty(Int_val(vfd)));
#else
  return Val_bool(isatty(Int_val(vfd)));
#endif
}

/* Set binary/text mode on fd (Windows/Cygwin only, no-op elsewhere).
   fd:int -> binary:bool -> unit */
CAMLprim value caml_stdlib_set_binary_mode(value vfd, value vbin) {
#if defined(_WIN32) || defined(__CYGWIN__)
  if (setmode(Int_val(vfd), Bool_val(vbin) ? O_BINARY : O_TEXT) == -1)
    caml_sys_error(NO_ARG);
#endif
  return Val_unit;
}

/* ---- Bigstring (char bigarray) access primitives ---- */

/* Unsafe get: bigstring -> int -> char (as int) */
CAMLprim value caml_bigstring_unsafe_get(value vba, value vidx) {
  unsigned char *data = (unsigned char *)Caml_ba_data_val(vba);
  return Val_int(data[Long_val(vidx)]);
}

/* Unsafe set: bigstring -> int -> char (as int) -> unit */
CAMLprim value caml_bigstring_unsafe_set(value vba, value vidx, value vc) {
  unsigned char *data = (unsigned char *)Caml_ba_data_val(vba);
  data[Long_val(vidx)] = Int_val(vc);
  return Val_unit;
}

/* Dimension of a 1-d bigarray: bigstring -> int */
CAMLprim value caml_bigstring_dim(value vba) {
  intnat d = Caml_ba_array_val(vba)->dim[0];
  return Val_long(d);
}

/* Create a 1-d char/c_layout bigarray of given length.
   Equivalent to Bigarray.Array1.create Bigarray.char Bigarray.c_layout len
   but callable without the Bigarray module being compiled yet.
   kind=Char=12, layout=C_layout=0 (per bigarray.h enum values).
   len:int -> bigstring */
CAMLprim value caml_ba_create_char_c1(value vlen) {
  intnat dim = Long_val(vlen);
  /* CAML_BA_CHAR=12, CAML_BA_C_LAYOUT=0 (see bigarray.h) */
  return caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED,
                            1, NULL, dim);
}

/* Return the number of rows (lines) of the terminal associated with fd,
   or -1 if fd is not a terminal or the size cannot be determined.
   fd:int -> int */
CAMLprim value caml_stdlib_terminfo_rows(value vfd) {
  return Val_int(caml_num_rows_fd(Int_val(vfd)));
}

/* Extract the file descriptor from a new-style fd-backed out_channel or in_channel.
   The new channel is an existential GADT block (tag=0):
     Out_ch { buf; ops; st; closed } where st = fd_state { fd; flags; binary; name }
   Field(vchannel, 2) is the state; Field(st, 0) is fd (as tagged int).
   Returns -1 if the channel is not fd-backed or has an unexpected layout.
   out_channel -> int  (or in_channel -> int) */
CAMLprim value caml_stdlib_channel_fd(value vchannel)
{
  if (Is_block(vchannel) && Tag_val(vchannel) == 0
      && Wosize_val(vchannel) >= 3) {
    value st = Field(vchannel, 2);
    if (Is_block(st) && Tag_val(st) == 0 && Wosize_val(st) >= 1) {
      value vfd = Field(st, 0);
      if (Is_long(vfd)) {
        return vfd;  /* already a tagged int */
      }
    }
  }
  return Val_int(-1);
}
