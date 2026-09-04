"""
MRD wire protocol identifiers and binary message structures.

Defines the message type identifiers used by :class:`server.connection.Connection`
to dispatch incoming/outgoing MRD messages, along with the ``struct.Struct``
objects used to pack and unpack their fixed-size binary fields.

Message identifiers
--------------------
MRD_MESSAGE_INT_ID_MIN : int
    Lower bound of the control message identifier range (0).
MRD_MESSAGE_CONFIG_FILE : int
    Identifier (1) for a configuration filename message.
MRD_MESSAGE_CONFIG_TEXT : int
    Identifier (2) for a configuration text message.
MRD_MESSAGE_METADATA_XML_TEXT : int
    Identifier (3) for the MRD XML header message.
MRD_MESSAGE_CLOSE : int
    Identifier (4) signalling the end of the data stream.
MRD_MESSAGE_TEXT : int
    Identifier (5) for an arbitrary text / logging message.
MRD_MESSAGE_ISMRMRD_ACQUISITION : int
    Identifier (1008) for a raw k-space acquisition message.
    Not currently supported by :class:`server.server.Server`.
MRD_MESSAGE_ISMRMRD_IMAGE : int
    Identifier (1022) for an image message.
MRD_MESSAGE_ISMRMRD_WAVEFORM : int
    Identifier (1026) for a waveform message.
    Not currently supported by :class:`server.server.Server`.

Binary structures
------------------
MrdMessageLength : struct.Struct
    Packs/unpacks the 4-byte (``<I``) message length field that precedes
    every variable-length MRD message body.
SIZEOF_MRD_MESSAGE_LENGTH : int
    Byte size of ``MrdMessageLength`` (4).
MrdMessageIdentifier : struct.Struct
    Packs/unpacks the 2-byte (``<H``) message identifier field.
SIZEOF_MRD_MESSAGE_IDENTIFIER : int
    Byte size of ``MrdMessageIdentifier`` (2).
MrdMessageConfigurationFile : struct.Struct
    Packs/unpacks the fixed 1024-byte (``<1024s``) configuration filename
    field used by ``MRD_MESSAGE_CONFIG_FILE``.
SIZEOF_MRD_MESSAGE_CONFIGURATION_FILE : int
    Byte size of ``MrdMessageConfigurationFile`` (1024).
MrdMessageAttribLength : struct.Struct
    Packs/unpacks the 8-byte (``<Q``) attribute length field preceding an
    image's serialised MetaAttributes.
SIZEOF_MRD_MESSAGE_ATTRIB_LENGTH : int
    Byte size of ``MrdMessageAttribLength`` (8).
"""

import struct

MRD_MESSAGE_INT_ID_MIN                             =    0 # CONTROL
MRD_MESSAGE_CONFIG_FILE                            =    1
MRD_MESSAGE_CONFIG_TEXT                            =    2
MRD_MESSAGE_METADATA_XML_TEXT                      =    3
MRD_MESSAGE_CLOSE                                  =    4
MRD_MESSAGE_TEXT                                   =    5
MRD_MESSAGE_ISMRMRD_ACQUISITION                    = 1008
MRD_MESSAGE_ISMRMRD_IMAGE                          = 1022
MRD_MESSAGE_ISMRMRD_WAVEFORM                       = 1026

MrdMessageLength = struct.Struct('<I')
SIZEOF_MRD_MESSAGE_LENGTH = len(MrdMessageLength.pack(0))

MrdMessageIdentifier = struct.Struct('<H')
SIZEOF_MRD_MESSAGE_IDENTIFIER = len(MrdMessageIdentifier.pack(0))

MrdMessageConfigurationFile = struct.Struct('<1024s')
SIZEOF_MRD_MESSAGE_CONFIGURATION_FILE = len(MrdMessageConfigurationFile.pack(b''))

MrdMessageAttribLength = struct.Struct('<Q')
SIZEOF_MRD_MESSAGE_ATTRIB_LENGTH = len(MrdMessageAttribLength.pack(0))
