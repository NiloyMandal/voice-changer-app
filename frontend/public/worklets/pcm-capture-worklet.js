class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 1024;
    this.buffer = new Float32Array(this.bufferSize);
    this.pointer = 0;
    this.poolCapacity = 8;
    this.freeBuffers = [];

    for (let i = 0; i < this.poolCapacity; i += 1) {
      this.freeBuffers.push(
        new ArrayBuffer(this.bufferSize * Int16Array.BYTES_PER_ELEMENT),
      );
    }

    this.port.onmessage = (event) => {
      const payload = event.data;
      if (!payload || payload.type !== "recycle") {
        return;
      }

      const recycled = payload.buffer;
      const expectedSize = this.bufferSize * Int16Array.BYTES_PER_ELEMENT;
      if (
        !(recycled instanceof ArrayBuffer) ||
        recycled.byteLength !== expectedSize
      ) {
        return;
      }

      if (this.freeBuffers.length < this.poolCapacity) {
        this.freeBuffers.push(recycled);
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const channel = input[0];
    if (!channel) {
      return true;
    }

    for (let i = 0; i < channel.length; i += 1) {
      this.buffer[this.pointer] = channel[i];
      this.pointer += 1;

      if (this.pointer >= this.bufferSize) {
        this.flush();
      }
    }

    return true;
  }

  flush() {
    const raw =
      this.freeBuffers.pop() ||
      new ArrayBuffer(this.bufferSize * Int16Array.BYTES_PER_ELEMENT);
    const pcm16 = new Int16Array(raw);

    for (let i = 0; i < this.bufferSize; i += 1) {
      const sample = Math.max(-1, Math.min(1, this.buffer[i]));
      pcm16[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }

    this.port.postMessage({ type: "pcm16", buffer: raw }, [raw]);
    this.pointer = 0;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
