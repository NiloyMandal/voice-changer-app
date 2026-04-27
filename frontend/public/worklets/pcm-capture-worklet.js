class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 2048;
    this.buffer = new Float32Array(this.bufferSize);
    this.pointer = 0;
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
    const pcm16 = new Int16Array(this.bufferSize);

    for (let i = 0; i < this.bufferSize; i += 1) {
      const sample = Math.max(-1, Math.min(1, this.buffer[i]));
      pcm16[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }

    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    this.pointer = 0;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
