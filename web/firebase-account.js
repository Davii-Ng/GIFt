import { firebaseConfig, AUTH_USERNAME_DOMAIN } from './firebase-config.js';

export function usernameIdentity(value) {
  const username = value.trim().toLowerCase();
  if (!/^[a-z0-9_]{3,24}$/.test(username)) {
    throw new Error('Username cần 3–24 ký tự, chỉ gồm chữ cái, chữ số và dấu gạch dưới.');
  }
  return { username, email: username + '@' + AUTH_USERNAME_DOMAIN };
}

export function accountError(error) {
  const messages = {
    'auth/configuration-not-found': 'Đăng nhập chưa được thiết lập trên Firebase. Chủ website cần bật Authentication và phương thức Email/Password.',
    'auth/operation-not-allowed': 'Phương thức đăng nhập chưa được bật. Chủ website cần bật Email/Password trong Firebase Authentication.',
    'auth/unauthorized-domain': 'Địa chỉ website này chưa được cho phép đăng nhập. Chủ website cần thêm hostname vào Firebase Authentication.',
    'auth/invalid-api-key': 'Cấu hình tài khoản chưa hợp lệ. Chủ website cần kiểm tra lại cấu hình Firebase.',
    'auth/invalid-credential': 'Tên đăng nhập hoặc mật khẩu không đúng.',
    'auth/user-not-found': 'Tên đăng nhập hoặc mật khẩu không đúng.',
    'auth/wrong-password': 'Tên đăng nhập hoặc mật khẩu không đúng.',
    'auth/email-already-in-use': 'Tên đăng nhập đã được sử dụng.',
    'auth/too-many-requests': 'Bạn thao tác quá nhiều lần. Vui lòng thử lại sau.',
    'auth/weak-password': 'Mật khẩu cần ít nhất 8 ký tự và đáp ứng chính sách đã cấu hình.',
    'auth/network-request-failed': 'Không thể kết nối. Vui lòng kiểm tra mạng và thử lại.',
    'permission-denied': 'Chưa thể truy cập dữ liệu đã lưu. Vui lòng thử lại.',
    'unavailable': 'Kết nối bị gián đoạn. Vui lòng thử lại.',
  };
  return messages[error?.code] || 'Thao tác chưa hoàn tất. Vui lòng thử lại.';
}

let sdkPromise;
async function sdk() {
  if (!firebaseConfig.apiKey || !firebaseConfig.authDomain || !firebaseConfig.projectId || !firebaseConfig.appId) {
    throw new Error('Tính năng tài khoản đang chờ cấu hình Firebase.');
  }
  if (!sdkPromise) {
    sdkPromise = (async () => {
      const base = 'https://www.gstatic.com/firebasejs/13.3.0/';
      const [app, auth, store] = await Promise.all([
        import(base + 'firebase-app.js'), import(base + 'firebase-auth.js'), import(base + 'firebase-firestore.js'),
      ]);
      const instance = app.initializeApp(firebaseConfig);
      const session = auth.getAuth(instance);
      await auth.setPersistence(session, auth.browserLocalPersistence);
      return { auth, store, session, db: store.getFirestore(instance) };
    })().catch(() => { sdkPromise = null; throw new Error('Không thể kết nối dịch vụ tài khoản. Vui lòng kiểm tra mạng và thử lại.'); });
  }
  return sdkPromise;
}

export async function observeAccount(onChange, onError) {
  const { auth, store, session, db } = await sdk();
  let epoch = 0;
  let stopSaved = () => {};
  const stopAuth = auth.onAuthStateChanged(session, user => {
    const revision = ++epoch;
    stopSaved();
    onChange({ user: user ? { id: user.uid, username: user.displayName || user.email.split('@')[0] } : null, saved: [], loading: Boolean(user) });
    if (!user) return;
    stopSaved = store.onSnapshot(store.collection(db, 'users', user.uid, 'favorites'), { includeMetadataChanges: true }, snapshot => {
      if (revision !== epoch || snapshot.metadata.hasPendingWrites) return;
      onChange({ user: { id: user.uid, username: user.displayName || user.email.split('@')[0] },
        saved: snapshot.docs.map(doc => doc.id), loading: false });
    }, error => {
      if (revision === epoch) onError(accountError(error));
    });
  }, () => onError('Không thể khôi phục phiên. Vui lòng thử tải lại trang.'));
  return () => { ++epoch; stopSaved(); stopAuth(); };
}

export async function authenticate(mode, value, password) {
  const { username, email } = usernameIdentity(value);
  if (password.length < 8 || password.length > 128) throw new Error('Mật khẩu cần 8–128 ký tự.');
  const { auth, session } = await sdk();
  try {
    const result = mode === 'register'
      ? await auth.createUserWithEmailAndPassword(session, email, password)
      : await auth.signInWithEmailAndPassword(session, email, password);
    if (mode === 'register' || !result.user.displayName) {
      try { await auth.updateProfile(result.user, { displayName: username }); }
      catch { return { user: { id: result.user.uid, username }, username, warning: 'Tài khoản đã sẵn sàng. Tên hiển thị sẽ được cập nhật ở lần đăng nhập sau.' }; }
    }
    return { user: { id: result.user.uid, username }, username };
  } catch (error) { throw new Error(accountError(error)); }
}

export async function logout() {
  const { auth, session } = await sdk();
  try { await auth.signOut(session); }
  catch (error) { throw new Error(accountError(error)); }
}

async function confirmed(operation) {
  let timer;
  try {
    return await Promise.race([operation, new Promise((_, reject) => {
      timer = setTimeout(() => reject({ code: 'unavailable' }), 15000);
    })]);
  } finally { clearTimeout(timer); }
}

export async function saveFavorite(uid, id, saved) {
  const { store, session, db } = await sdk();
  if (session.currentUser?.uid !== uid) throw new Error('Phiên đăng nhập đã thay đổi. Vui lòng thử lại.');
  const ref = store.doc(db, 'users', uid, 'favorites', id);
  try {
    await confirmed(store.getDocFromServer(ref));
    if (session.currentUser?.uid !== uid) throw { code: 'permission-denied' };
    if (saved) await confirmed(store.setDoc(ref, { gifId: id }));
    else await confirmed(store.deleteDoc(ref));
  } catch (error) { throw new Error(accountError(error)); }
}

export async function accountToken(expectedUid) {
  const { auth, session } = await sdk();
  if (!session.currentUser) throw new Error('Log in to create and manage your GIFs.');
  if (expectedUid && session.currentUser.uid !== expectedUid) throw new Error('Your account changed. Please try again.');
  const token = await auth.getIdToken(session.currentUser);
  if (expectedUid && session.currentUser?.uid !== expectedUid) throw new Error('Your account changed. Please try again.');
  return token;
}

export async function syncContribution(gif, remove = false, expectedUid) {
  const { store, session, db } = await sdk();
  const uid = session.currentUser?.uid;
  if (!uid) throw new Error('Log in to sync your contribution.');
  if (expectedUid && uid !== expectedUid) throw new Error('Your account changed. Please try again.');
  const ref = store.doc(db, 'users', uid, 'contributions', gif.id);
  try {
    if (remove) await confirmed(store.deleteDoc(ref));
    else await confirmed(store.setDoc(ref, {
      gifId: gif.id, title: gif.title, description: gif.description,
      visibility: gif.visibility, status: gif.status,
      createdAt: gif.created, updatedAt: gif.updated,
    }));
  } catch (error) { throw new Error(accountError(error)); }
}
