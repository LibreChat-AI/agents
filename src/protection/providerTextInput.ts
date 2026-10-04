const inputs = new WeakMap<object, { prototype: object | null; descriptors: PropertyDescriptorMap }>();

/** Internal provenance for the SDK's pure instruction transform, not a host capability. */
export function markProviderTextInput<T extends object>(input: T): T {
  inputs.set(input, { prototype: Object.getPrototypeOf(input), descriptors: Object.getOwnPropertyDescriptors(input) });
  return input;
}

export function isProviderTextInput(input: object): boolean {
  const original = inputs.get(input);
  if (original == null || original.prototype !== Object.getPrototypeOf(input)) return false;
  const current: Partial<PropertyDescriptorMap> = Object.getOwnPropertyDescriptors(input);
  const keys = Object.keys(original.descriptors);
  return keys.length === Object.keys(current).length && keys.every((key) => {
    const before = original.descriptors[key];
    const after = current[key];
    return after != null && before.value === after.value && before.get === after.get &&
      before.set === after.set && before.writable === after.writable &&
      before.enumerable === after.enumerable && before.configurable === after.configurable;
  });
}
